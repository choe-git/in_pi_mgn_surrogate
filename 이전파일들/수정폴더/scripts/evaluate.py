from __future__ import annotations

import argparse
from pathlib import Path

import torch

from gnn_surrogate.data import (
    apply_velocity_boundary,
    CaseCache,
    fluid_node_mask,
    load_graph_sample,
    list_h5_files,
    split_files,
)
from gnn_surrogate.metrics import mean_and_sd, rmse_mm_s
from gnn_surrogate.model import MeshGraphNet, build_node_features, node_feature_dim
from gnn_surrogate.train_utils import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rollout-steps", type=int, default=50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-test-cases", type=int, default=None)
    parser.add_argument("--clamp-inlet", action="store_true", default=True)
    parser.add_argument("--no-clamp-inlet", action="store_false", dest="clamp_inlet")
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    args = ckpt["args"]
    model = MeshGraphNet(
        node_input_dim=node_feature_dim(args["model_variant"]),
        latent_dim=args["latent_dim"],
        hidden_dim=args["hidden_dim"],
        message_passing_steps=args["message_passing_steps"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    mean = torch.tensor(ckpt["velocity_stats"]["mean"], dtype=torch.float32, device=device)
    std = torch.tensor(ckpt["velocity_stats"]["std"], dtype=torch.float32, device=device).clamp_min(1e-6)
    return model, args, mean, std, ckpt


@torch.no_grad()
def one_step_case(model, path, cache, model_variant, mean, std, device, clamp_inlet: bool) -> float:
    values = []
    for t in range(1, 79):
        sample = load_graph_sample(path, t, cache, device=device)
        features = build_node_features(sample, model_variant, mean, std)
        pred_u = sample.current_u + model(features, sample.edge_index, sample.edge_attr) * std
        pred_u = apply_velocity_boundary(pred_u, sample, clamp_inlet=clamp_inlet)
        values.append(float(rmse_mm_s(pred_u, sample.target_u, fluid_node_mask(sample)).cpu()))
    return sum(values) / max(len(values), 1)


@torch.no_grad()
def rollout_case(model, path, cache, model_variant, mean, std, device, steps: int, clamp_inlet: bool) -> float:
    sample = load_graph_sample(path, 1, cache, device=device)
    prev_u = sample.prev_u
    current_u = sample.current_u
    rmses = []
    for offset in range(steps):
        t = 1 + offset
        if t >= 79:
            break
        sample = load_graph_sample(path, t, cache, device=device)
        sample.prev_u = prev_u
        sample.current_u = current_u
        features = build_node_features(sample, model_variant, mean, std)
        pred_u = current_u + model(features, sample.edge_index, sample.edge_attr) * std
        pred_u = apply_velocity_boundary(pred_u, sample, clamp_inlet=clamp_inlet)
        rmses.append(float(rmse_mm_s(pred_u, sample.target_u, fluid_node_mask(sample)).cpu()))
        prev_u, current_u = current_u, pred_u
    return sum(rmses) / max(len(rmses), 1)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model, train_args, mean, std, ckpt = load_model(Path(args.checkpoint), device)
    files = list_h5_files(args.data_dir)
    _, test_files = split_files(files, test_files=ckpt["split"]["test"])
    if args.max_test_cases:
        test_files = test_files[: args.max_test_cases]
    cache = CaseCache(train_args.get("boundary_percentile", 2.0))

    one_step = []
    rollout = []
    for path in test_files:
        r1 = one_step_case(
            model,
            path,
            cache,
            train_args["model_variant"],
            mean,
            std,
            device,
            args.clamp_inlet,
        )
        r50 = rollout_case(
            model,
            path,
            cache,
            train_args["model_variant"],
            mean,
            std,
            device,
            args.rollout_steps,
            args.clamp_inlet,
        )
        one_step.append(r1)
        rollout.append(r50)
        print(f"{path.name}: 1-RMSE={r1:.4f} {args.rollout_steps}-RMSE={r50:.4f}")

    m1, s1 = mean_and_sd(one_step)
    mr, sr = mean_and_sd(rollout)
    print(f"MEAN 1-RMSE={m1:.4f} SD={s1:.4f}")
    print(f"MEAN {args.rollout_steps}-RMSE={mr:.4f} SD={sr:.4f}")


if __name__ == "__main__":
    main()
