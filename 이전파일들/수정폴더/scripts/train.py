from __future__ import annotations
from datetime import datetime

import argparse
import math
import time
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

import torch
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gnn_surrogate.data import (
    apply_velocity_boundary,
    CaseCache,
    fluid_node_mask,
    learned_node_mask,
    load_graph_sample,
    list_h5_files,
    make_training_index,
    save_split,
    split_files,
)
from gnn_surrogate.metrics import rmse_mm_s
from gnn_surrogate.model import (
    MeshGraphNet,
    add_training_noise,
    build_node_features,
    node_feature_dim,
)
from gnn_surrogate.physics import physics_losses
from gnn_surrogate.train_utils import (
    compute_velocity_stats,
    make_lr,
    resolve_device,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-variant", choices=["mgn", "pi-mgn", "in-mgn", "in-pi-mgn"], default="in-pi-mgn")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--message-passing-steps", type=int, default=15)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-min", type=float, default=1e-7)
    parser.add_argument("--lr-decay-start-epoch", type=int, default=16)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--noise-std", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--test-count", type=int, default=10)
    parser.add_argument("--test-files", nargs="*", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--boundary-percentile", type=float, default=2.0)
    parser.add_argument("--stats-samples", type=int, default=128)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-cases", type=int, default=None)
    parser.add_argument("--max-eval-steps", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--num-workers-note", default="batch_size_is_one_by_design")
    return parser.parse_args()


def train_one_sample(model, sample, args, velocity_mean, velocity_std) -> tuple[torch.Tensor, dict[str, float]]:
    current_u = add_training_noise(sample, velocity_std, args.noise_std)
    features = build_node_features(
        sample,
        args.model_variant,
        velocity_mean,
        velocity_std,
        current_u=current_u,
    )
    pred_delta_norm = model(features, sample.edge_index, sample.edge_attr)
    pred_delta = pred_delta_norm * velocity_std
    pred_u = apply_velocity_boundary(current_u + pred_delta, sample)
    mask = learned_node_mask(sample)
    data_loss = torch.mean((pred_u[mask] - sample.target_u[mask]).square())
    losses = {"data": float(data_loss.detach().cpu())}
    if "pi" in args.model_variant:
        phys = physics_losses(
            pred_u,
            sample.target_u,
            current_u,
            sample.points,
            sample.edge_index,
            mask,
        )
        loss = 0.5 * data_loss + (phys["continuity"] + phys["convection"] + phys["viscosity"]) / 6.0
        losses.update({k: float(v.detach().cpu()) for k, v in phys.items()})
    else:
        loss = data_loss
    losses["rmse"] = float(rmse_mm_s(pred_u.detach(), sample.target_u, mask).cpu())
    return loss, losses


@torch.no_grad()
def evaluate_one_step(model, test_files, cache, args, velocity_mean, velocity_std, device) -> float:
    model.eval()
    rmses = []
    files = test_files[: args.max_test_cases] if args.max_test_cases else test_files
    for path in files:
        steps = range(1, 79)
        if args.max_eval_steps:
            steps = range(1, min(79, 1 + args.max_eval_steps))
        for t in steps:
            sample = load_graph_sample(path, t, cache, device=device)
            features = build_node_features(sample, args.model_variant, velocity_mean, velocity_std)
            pred_delta = model(features, sample.edge_index, sample.edge_attr) * velocity_std
            pred_u = apply_velocity_boundary(sample.current_u + pred_delta, sample)
            mask = fluid_node_mask(sample)
            rmses.append(float(rmse_mm_s(pred_u, sample.target_u, mask).detach().cpu()))
    model.train()
    return sum(rmses) / max(len(rmses), 1)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir = output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list_h5_files(args.data_dir)
    train_files, test_files = split_files(files, args.seed, args.test_count, args.test_files)
    save_split(output_dir / "split.json", train_files, test_files)
    save_json(output_dir / "args.json", vars(args))

    cache = CaseCache(args.boundary_percentile)
    train_index = make_training_index(train_files)
    if args.max_train_samples:
        train_index = train_index[: args.max_train_samples]

    stats = compute_velocity_stats(train_index, cache, max_samples=args.stats_samples)
    save_json(output_dir / "velocity_stats.json", stats)
    device = resolve_device(args.device)
    velocity_mean = torch.tensor(stats["mean"], dtype=torch.float32, device=device)
    velocity_std = torch.tensor(stats["std"], dtype=torch.float32, device=device).clamp_min(1e-6)

    model = MeshGraphNet(
        node_input_dim=node_feature_dim(args.model_variant),
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        message_passing_steps=args.message_passing_steps,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_rmse = math.inf
    global_step = 0
    for epoch in range(args.epochs):
        lr = make_lr(optimizer, args.lr, args.lr_min, epoch, args.lr_decay_start_epoch, args.epochs)
        epoch_start = time.time()
        running = []
        order = torch.randperm(len(train_index)).tolist()
        pbar = tqdm(order, desc=f"epoch {epoch + 1}/{args.epochs} lr={lr:.2e}")
        for idx in pbar:
            path, t = train_index[idx]
            sample = load_graph_sample(path, t, cache, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss, losses = train_one_sample(model, sample, args, velocity_mean, velocity_std)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1
            running.append(losses["rmse"])
            writer.add_scalar("train/loss", loss.detach().item(), epoch)
            writer.add_scalar("train/1-rmse", float(losses["rmse"]), epoch)
            if len(running) >= 20:
                pbar.set_postfix(train_rmse=f"{sum(running[-20:]) / 20:.3f}")

        eval_rmse = float("nan")
        torch.save(
            {
                "model_state": model.state_dict(),
                "args": vars(args),
                "velocity_stats": stats,
                "split": {
                    "train": [p.name for p in train_files],
                    "test": [p.name for p in test_files],
                },
            },
            output_dir / "last.pt",
        )
        if (epoch + 1) % args.eval_every == 0:
            eval_rmse = evaluate_one_step(model, test_files, cache, args, velocity_mean, velocity_std, device)
            #new
            writer.add_scalar("eval_1rmse", eval_rmse, epoch)
            if eval_rmse < best_rmse:
                best_rmse = eval_rmse
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "args": vars(args),
                        "velocity_stats": stats,
                        "split": {
                            "train": [p.name for p in train_files],
                            "test": [p.name for p in test_files],
                        },
                    },
                    output_dir / "best.pt",
                )
        print(
            f"epoch={epoch + 1} train_rmse={sum(running) / max(len(running), 1):.4f} "
            f"eval_1rmse={eval_rmse:.4f} best={best_rmse:.4f} "
            f"seconds={time.time() - epoch_start:.1f}"
        )

if __name__ == "__main__":
    main()
