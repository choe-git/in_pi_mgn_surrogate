from __future__ import annotations
from datetime import datetime

import argparse
import math
import time
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

import torch
from tqdm import tqdm

from gnn_surrogate.data import (
    apply_velocity_boundary,
    CaseCache,
    learned_node_mask,
    load_graph_sample,
    list_h5_files,
    make_training_index,
    save_split_three_way,
    split_files_three_way,
)
from gnn_surrogate.metrics import rmse_mm_s
from gnn_surrogate.inlet_profile import CanonicalInletProfile, default_output_path
from gnn_surrogate.model import (
    MeshGraphNet,
    augment_training_inputs,
    build_node_features,
    node_feature_dim,
)
from gnn_surrogate.physics import physics_losses
from gnn_surrogate.train_utils import (
    compute_acceleration_stats,
    compute_velocity_stats,
    make_lr,
    normalize_acceleration_mode,
    normalize_output_scale,
    parse_optional_sample_count,
    resolve_device,
    save_json,
    set_seed,
)
FIXED_SEED = 2026
VALIDATION_ROLLOUT_STEPS = 50



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--split-csv",
        required=True,
        help="CSV with data_name,train,val,test binary one-hot columns",
    )
    parser.add_argument(
        "--inlet-profile",
        default=str(default_output_path()),
        help="Canonical inlet profile .npz used by in-mgn variants",
    )
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
    parser.add_argument(
        "--velocity-augmentations",
        nargs="*",
        choices=["1", "2", "3"],
        default=["1", "2", "3"],
        metavar="N",
        help="1=temporal noise, 2=spatially smooth noise, 3=magnitude jitter; pass the flag alone to disable all.",
    )
    parser.add_argument("--temporal-noise-correlation", type=float, default=0.8)
    parser.add_argument("--spatial-noise-smoothing-steps", type=int, default=2)
    parser.add_argument("--magnitude-jitter-std", type=float, default=0.02)
    parser.add_argument("--val-count", type=int, default=5)
    parser.add_argument("--test-count", type=int, default=5)
    parser.add_argument("--val-files", nargs="*", default=None)
    parser.add_argument("--test-files", nargs="*", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--boundary-percentile", type=float, default=2.0)
    parser.add_argument("--stats-samples", type=parse_optional_sample_count, default=None)
    parser.add_argument("--stats-sampling", choices=["head", "uniform", "random"], default="uniform")
    parser.add_argument("--output-scale", default="acceleration")
    parser.add_argument("--acceleration-mode", "--accel-mode", dest="acceleration_mode", default="acceleration")
    parser.add_argument("--physics-operator", choices=["legacy", "gradient"], default="gradient")
    parser.add_argument("--continuity-target", choices=["zero", "match"], default="zero")
    parser.add_argument("--data-loss-weight", type=float, default=0.5)
    parser.add_argument("--continuity-weight", type=float, default=1.0 / 6.0)
    parser.add_argument("--convection-weight", type=float, default=1.0 / 6.0)
    parser.add_argument("--viscosity-weight", type=float, default=1.0 / 6.0)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-cases", "--max-test-cases", dest="max_val_cases", type=int, default=None)
    parser.add_argument("--max-eval-steps", type=int, default=None)
    parser.add_argument("--eval-rollout-steps", type=int, default=0)
    parser.add_argument("--best-rmse-steps", type=int, default=50)
    parser.add_argument("--num-workers-note", default="batch_size_is_one_by_design")
    args = parser.parse_args()
    args.seed = FIXED_SEED
    args.eval_domain = "whole"
    args.validation_rollout_steps = VALIDATION_ROLLOUT_STEPS
    args.output_scale = normalize_output_scale(args.output_scale)
    args.acceleration_mode = normalize_acceleration_mode(args.acceleration_mode)
    if args.best_rmse_steps < 1:
        raise ValueError("--best-rmse-steps must be >= 1")
    if not 0.0 <= args.temporal_noise_correlation <= 1.0:
        raise ValueError("--temporal-noise-correlation must be in [0, 1]")
    if args.spatial_noise_smoothing_steps < 0:
        raise ValueError("--spatial-noise-smoothing-steps must be non-negative")
    if args.magnitude_jitter_std < 0.0:
        raise ValueError("--magnitude-jitter-std must be non-negative")
    return args


def whole_geometry_mask(sample) -> torch.Tensor:
    return torch.ones_like(sample.wall_mask, dtype=torch.bool)


def checkpoint_payload(model, args, velocity_stats, output_stats, split) -> dict:
    return {
        "model_state": model.state_dict(),
        "args": vars(args),
        "velocity_stats": velocity_stats,
        "output_stats": output_stats,
        "split": split,
    }


def prediction_loss(pred_u, sample, current_u, args) -> tuple[torch.Tensor, dict[str, float]]:
    mask = learned_node_mask(sample)
    data_loss = torch.mean((pred_u[mask] - sample.target_u[mask]).square())
    losses = {"data": float(data_loss.detach().cpu())}
    if "pi" not in args.model_variant:
        return data_loss, losses

    phys = physics_losses(
        pred_u,
        sample.target_u,
        current_u,
        sample.points,
        sample.edge_index,
        mask,
        operator_mode=args.physics_operator,
        continuity_target=args.continuity_target,
    )
    loss = (
        args.data_loss_weight * data_loss
        + args.continuity_weight * phys["continuity"]
        + args.convection_weight * phys["convection"]
        + args.viscosity_weight * phys["viscosity"]
    )
    losses.update({k: float(v.detach().cpu()) for k, v in phys.items()})
    return loss, losses


def train_one_sample(
    model,
    sample,
    args,
    velocity_mean,
    velocity_std,
    output_mean,
    output_std,
) -> tuple[torch.Tensor, dict[str, float]]:
    augmentation = augment_training_inputs(
        sample,
        velocity_std,
        args.velocity_augmentations,
        args.noise_std,
        args.temporal_noise_correlation,
        args.spatial_noise_smoothing_steps,
        args.magnitude_jitter_std,
    )
    features = build_node_features(
        sample,
        args.model_variant,
        velocity_mean,
        velocity_std,
        current_u=augmentation.current_u,
        prev_u=augmentation.prev_u,
        inflow_context=augmentation.inflow_context,
        acceleration_mode=args.acceleration_mode,
    )
    pred_acceleration_norm = model(features, sample.edge_index, sample.edge_attr)
    pred_acceleration = pred_acceleration_norm * output_std + output_mean
    pred_u = apply_velocity_boundary(
        augmentation.current_u + pred_acceleration,
        sample,
        clamp_inlet=False,
    )
    mask = learned_node_mask(sample)
    loss, losses = prediction_loss(pred_u, sample, augmentation.current_u, args)
    losses["rmse"] = float(rmse_mm_s(pred_u.detach(), sample.target_u, mask).cpu())
    return loss, losses


@torch.no_grad()
def evaluate_one_step(model, val_files, cache, args, velocity_mean, velocity_std, output_mean, output_std, device) -> tuple[float, float]:
    model.eval()
    rmses = []
    losses = []
    files = val_files[: args.max_val_cases] if args.max_val_cases else val_files
    for path in files:
        steps = range(1, 79)
        if args.max_eval_steps:
            steps = range(1, min(79, 1 + args.max_eval_steps))
        for t in steps:
            sample = load_graph_sample(path, t, cache, device=device)
            features = build_node_features(
                sample,
                args.model_variant,
                velocity_mean,
                velocity_std,
                acceleration_mode=args.acceleration_mode,
            )
            pred_acceleration = model(features, sample.edge_index, sample.edge_attr) * output_std + output_mean
            pred_u = apply_velocity_boundary(
                sample.current_u + pred_acceleration,
                sample,
                clamp_inlet=False,
            )
            loss, _ = prediction_loss(pred_u, sample, sample.current_u, args)
            mask = whole_geometry_mask(sample)
            rmses.append(float(rmse_mm_s(pred_u, sample.target_u, mask).detach().cpu()))
            losses.append(float(loss.detach().cpu()))
    model.train()
    return sum(rmses) / max(len(rmses), 1), sum(losses) / max(len(losses), 1)


@torch.no_grad()
def evaluate_rollout(model, val_files, cache, args, velocity_mean, velocity_std, output_mean, output_std, device, rollout_steps: int | None = None) -> float:
    model.eval()
    rmses = []
    files = val_files[: args.max_val_cases] if args.max_val_cases else val_files
    rollout_steps = args.eval_rollout_steps if rollout_steps is None else rollout_steps
    for path in files:
        sample = load_graph_sample(path, 1, cache, device=device)
        prev_u = sample.prev_u
        current_u = sample.current_u
        for offset in range(rollout_steps):
            t = 1 + offset
            if t >= 79:
                break
            sample = load_graph_sample(path, t, cache, device=device)
            sample.prev_u = prev_u
            sample.current_u = current_u
            features = build_node_features(
                sample,
                args.model_variant,
                velocity_mean,
                velocity_std,
                acceleration_mode=args.acceleration_mode,
            )
            pred_acceleration = model(features, sample.edge_index, sample.edge_attr) * output_std + output_mean
            pred_u = apply_velocity_boundary(
                current_u + pred_acceleration,
                sample,
                clamp_inlet=False,
            )
            mask = whole_geometry_mask(sample)
            rmses.append(float(rmse_mm_s(pred_u, sample.target_u, mask).detach().cpu()))
            prev_u, current_u = current_u, pred_u
    model.train()
    return sum(rmses) / max(len(rmses), 1)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    inlet_profile = None
    if "in" in args.model_variant:
        inlet_profile = CanonicalInletProfile.load(args.inlet_profile)
        print(f"inlet_profile={Path(args.inlet_profile).expanduser().resolve()}")
    output_dir = Path(args.output_dir)
    output_dir = output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(output_dir), max_queue=1)
    print(f"tensorboard_log_dir={output_dir.resolve()}")

    files = list_h5_files(args.data_dir)
    train_files, val_files, test_files = split_files_three_way(
        files,
        seed=args.seed,
        val_count=args.val_count,
        test_count=args.test_count,
        val_files=args.val_files,
        test_files=args.test_files,
        split_csv=args.split_csv,
    )
    split = save_split_three_way(
        output_dir / "split.json",
        train_files,
        val_files,
        test_files,
        seed=None,
        split_csv=args.split_csv,
    )
    save_json(output_dir / "args.json", vars(args))
    print(
        f"split_csv={Path(args.split_csv).expanduser().resolve()} "
        f"train={len(train_files)} val={len(val_files)} test={len(test_files)}"
    )

    cache = CaseCache(args.boundary_percentile, inlet_profile=inlet_profile)
    train_index = make_training_index(train_files)
    if args.max_train_samples:
        train_index = train_index[: args.max_train_samples]

    stats = compute_velocity_stats(
        train_index,
        cache,
        max_samples=args.stats_samples,
        sampling=args.stats_sampling,
        seed=args.seed,
    )
    save_json(output_dir / "velocity_stats.json", stats)
    if args.output_scale == "acceleration":
        output_stats = compute_acceleration_stats(
            train_index,
            cache,
            max_samples=args.stats_samples,
            sampling=args.stats_sampling,
            seed=args.seed,
        )
        save_json(output_dir / "acceleration_stats.json", output_stats)
    else:
        output_stats = {
            "mean": [0.0, 0.0, 0.0],
            "std": stats["std"],
            "value": "velocity",
            "samples": stats.get("samples"),
            "sampling": stats.get("sampling"),
        }
    save_json(output_dir / "output_stats.json", output_stats)
    device = resolve_device(args.device)
    velocity_mean = torch.tensor(stats["mean"], dtype=torch.float32, device=device)
    velocity_std = torch.tensor(stats["std"], dtype=torch.float32, device=device).clamp_min(1e-6)
    output_mean = torch.tensor(output_stats["mean"], dtype=torch.float32, device=device)
    output_std = torch.tensor(output_stats["std"], dtype=torch.float32, device=device).clamp_min(1e-6)

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
            loss, losses = train_one_sample(
                model,
                sample,
                args,
                velocity_mean,
                velocity_std,
                output_mean,
                output_std,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1
            running.append(losses["rmse"])
            writer.add_scalar("train/loss", loss.detach().item(), global_step)
            writer.add_scalar("train/1-rmse", float(losses["rmse"]), global_step)
            if len(running) >= 20:
                pbar.set_postfix(train_rmse=f"{sum(running[-20:]) / 20:.3f}")

        torch.save(
            checkpoint_payload(model, args, stats, output_stats, split),
            output_dir / "last.pt",
        )
        eval_rmse, validation_loss = evaluate_one_step(
            model,
            val_files,
            cache,
            args,
            velocity_mean,
            velocity_std,
            output_mean,
            output_std,
            device,
        )
        writer.add_scalar("eval/1_step_rmse", eval_rmse, epoch + 1)
        writer.add_scalar("eval/validation_loss", validation_loss, epoch + 1)
        rollout_rmses = {
            VALIDATION_ROLLOUT_STEPS: evaluate_rollout(
                model,
                val_files,
                cache,
                args,
                velocity_mean,
                velocity_std,
                output_mean,
                output_std,
                device,
                rollout_steps=VALIDATION_ROLLOUT_STEPS,
            )
        }
        validation_50_rmse = rollout_rmses[VALIDATION_ROLLOUT_STEPS]
        writer.add_scalar("eval/50_rollout_rmse", validation_50_rmse, epoch + 1)
        test_50_rmse = evaluate_rollout(
            model,
            test_files,
            cache,
            args,
            velocity_mean,
            velocity_std,
            output_mean,
            output_std,
            device,
            rollout_steps=VALIDATION_ROLLOUT_STEPS,
        )
        writer.add_scalar("test/mean-50-rmse", test_50_rmse, epoch + 1)
        if args.eval_rollout_steps > 0 and args.eval_rollout_steps not in rollout_rmses:
            rollout_rmses[args.eval_rollout_steps] = evaluate_rollout(
                model,
                val_files,
                cache,
                args,
                velocity_mean,
                velocity_std,
                output_mean,
                output_std,
                device,
                rollout_steps=args.eval_rollout_steps,
            )
        if args.best_rmse_steps == 1:
            selection_rmse = eval_rmse
            best_label = "1rmse"
        else:
            if args.best_rmse_steps not in rollout_rmses:
                rollout_rmses[args.best_rmse_steps] = evaluate_rollout(
                    model,
                    val_files,
                    cache,
                    args,
                    velocity_mean,
                    velocity_std,
                    output_mean,
                    output_std,
                    device,
                    rollout_steps=args.best_rmse_steps,
                )
            selection_rmse = rollout_rmses[args.best_rmse_steps]
            best_label = f"{args.best_rmse_steps}rmse"
        writer.add_scalar("eval/selection_rmse", selection_rmse, epoch + 1)
        if selection_rmse < best_rmse:
            best_rmse = selection_rmse
            torch.save(
                checkpoint_payload(model, args, stats, output_stats, split),
                output_dir / "best.pt",
            )
        print(
            f"epoch={epoch + 1} train_rmse={sum(running) / max(len(running), 1):.4f} "
            f"val_loss={validation_loss:.4g} "
            f"val_1rmse_whole={eval_rmse:.4f} val_50rmse_whole={validation_50_rmse:.4f} "
            f"test_mean_50rmse_whole={test_50_rmse:.4f} "
            f"val_selection_rmse={selection_rmse:.4f} best_{best_label}_whole={best_rmse:.4f} "
            f"seconds={time.time() - epoch_start:.1f}"
        )
    writer.close()

if __name__ == "__main__":
    main()
