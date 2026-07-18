from __future__ import annotations

import argparse
from pathlib import Path

import torch

from gnn_surrogate.data import (
    apply_velocity_boundary,
    CaseCache,
    load_graph_sample,
    list_h5_files,
    resolve_named_files,
)
from gnn_surrogate.metrics import mean_and_sd, rmse_mm_s
from gnn_surrogate.model import MeshGraphNet, build_node_features, node_feature_dim
from gnn_surrogate.train_utils import normalize_acceleration_mode, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rollout-steps", type=int, default=50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--split", choices=["test", "val", "train"], default="test")
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
    output_stats = ckpt.get(
        "output_stats",
        {"mean": [0.0, 0.0, 0.0], "std": ckpt["velocity_stats"]["std"]},
    )
    output_mean = torch.tensor(output_stats["mean"], dtype=torch.float32, device=device)
    output_std = torch.tensor(output_stats["std"], dtype=torch.float32, device=device).clamp_min(1e-6)
    return model, args, mean, std, output_mean, output_std, ckpt


def checkpoint_acceleration_mode(train_args: dict) -> str:
    return normalize_acceleration_mode(
        train_args.get("acceleration_mode", train_args.get("accel_mode", "physical_acceleration"))
    )


def whole_geometry_mask(sample) -> torch.Tensor:
    return torch.ones_like(sample.wall_mask, dtype=torch.bool)


def predict_next_velocity(
    model,
    sample,
    model_variant,
    acceleration_mode,
    mean,
    std,
    output_mean,
    output_std,
    clamp_inlet: bool,
) -> torch.Tensor:
    features = build_node_features(
        sample,
        model_variant,
        mean,
        std,
        acceleration_mode=acceleration_mode,
    )
    pred_acceleration = model(features, sample.edge_index, sample.edge_attr) * output_std + output_mean
    return apply_velocity_boundary(
        sample.current_u + pred_acceleration,
        sample,
        clamp_inlet=clamp_inlet,
    )


@torch.no_grad()
def one_step_case(model, path, cache, model_variant, train_args, mean, std, output_mean, output_std, device, clamp_inlet: bool) -> float:
    rmses = []
    acceleration_mode = checkpoint_acceleration_mode(train_args)
    for t in range(1, 79):
        sample = load_graph_sample(path, t, cache, device=device)
        pred_u = predict_next_velocity(
            model,
            sample,
            model_variant,
            acceleration_mode,
            mean,
            std,
            output_mean,
            output_std,
            clamp_inlet,
        )
        rmses.append(float(rmse_mm_s(pred_u, sample.target_u, whole_geometry_mask(sample)).cpu()))
    return sum(rmses) / max(len(rmses), 1)


@torch.no_grad()
def rollout_case(model, path, cache, model_variant, train_args, mean, std, output_mean, output_std, device, steps: int, clamp_inlet: bool) -> float:
    sample = load_graph_sample(path, 1, cache, device=device)
    prev_u = sample.prev_u
    current_u = sample.current_u
    rmses = []
    acceleration_mode = checkpoint_acceleration_mode(train_args)
    for offset in range(steps):
        t = 1 + offset
        if t >= 79:
            break
        sample = load_graph_sample(path, t, cache, device=device)
        sample.prev_u = prev_u
        sample.current_u = current_u
        pred_u = predict_next_velocity(
            model,
            sample,
            model_variant,
            acceleration_mode,
            mean,
            std,
            output_mean,
            output_std,
            clamp_inlet,
        )
        rmses.append(float(rmse_mm_s(pred_u, sample.target_u, whole_geometry_mask(sample)).cpu()))
        prev_u, current_u = current_u, pred_u
    return sum(rmses) / max(len(rmses), 1)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model, train_args, mean, std, output_mean, output_std, ckpt = load_model(Path(args.checkpoint), device)
    files = list_h5_files(args.data_dir)
    split_payload = ckpt["split"]
    if args.split not in split_payload:
        available = ", ".join(sorted(split_payload))
        raise KeyError(f"Checkpoint split has no '{args.split}' split. Available splits: {available}")
    evaluation_files = resolve_named_files(files, split_payload[args.split], args.split)
    if args.max_test_cases:
        evaluation_files = evaluation_files[: args.max_test_cases]
    print(f"evaluating split={args.split} cases={len(evaluation_files)} domain=whole")
    cache = CaseCache(train_args.get("boundary_percentile", 2.0))

    one_step_rmses = []
    rollout_rmses = []
    for path in evaluation_files:
        one_step = one_step_case(
            model,
            path,
            cache,
            train_args["model_variant"],
            train_args,
            mean,
            std,
            output_mean,
            output_std,
            device,
            args.clamp_inlet,
        )
        rollout = rollout_case(
            model,
            path,
            cache,
            train_args["model_variant"],
            train_args,
            mean,
            std,
            output_mean,
            output_std,
            device,
            args.rollout_steps,
            args.clamp_inlet,
        )
        one_step_rmses.append(one_step)
        rollout_rmses.append(rollout)
        print(
            f"{path.name}: 1-RMSE-whole={one_step:.4f} "
            f"{args.rollout_steps}-RMSE-whole={rollout:.4f}"
        )

    one_step_mean, one_step_sd = mean_and_sd(one_step_rmses)
    rollout_mean, rollout_sd = mean_and_sd(rollout_rmses)
    print(f"MEAN 1-RMSE-whole={one_step_mean:.4f} SD={one_step_sd:.4f}")
    print(f"MEAN {args.rollout_steps}-RMSE-whole={rollout_mean:.4f} SD={rollout_sd:.4f}")


if __name__ == "__main__":
    main()
