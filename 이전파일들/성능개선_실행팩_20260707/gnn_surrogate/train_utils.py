from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .data import CaseCache, learned_node_mask, load_graph_sample


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def select_stat_samples(
    train_index: Sequence[tuple[Path, int]],
    max_samples: int | None = 128,
    mode: str = "uniform",
    seed: int = 2026,
) -> list[tuple[Path, int]]:
    index = list(train_index)
    if max_samples is None or max_samples >= len(index):
        return index
    if max_samples <= 0:
        return []
    if mode == "head":
        return index[:max_samples]
    if mode == "random":
        rng = np.random.default_rng(seed)
        selected = sorted(rng.choice(len(index), size=max_samples, replace=False).tolist())
        return [index[i] for i in selected]
    if mode != "uniform":
        raise ValueError(f"Unknown stat sampling mode: {mode}")
    selected = np.linspace(0, len(index) - 1, num=max_samples, dtype=np.int64).tolist()
    return [index[i] for i in selected]


def compute_vector_stats(
    train_index: Sequence[tuple[Path, int]],
    cache: CaseCache,
    value: str = "velocity",
    max_samples: int | None = 128,
    sampling: str = "uniform",
    seed: int = 2026,
) -> dict[str, object]:
    train_index = select_stat_samples(train_index, max_samples, sampling, seed)
    total = torch.zeros(3, dtype=torch.float64)
    total_sq = torch.zeros(3, dtype=torch.float64)
    count = 0
    for path, t in train_index:
        sample = load_graph_sample(path, t, cache, device="cpu")
        if value == "velocity":
            mask = ~sample.wall_mask
            values = sample.current_u[mask].double()
        elif value == "delta":
            mask = learned_node_mask(sample)
            values = sample.target_delta[mask].double()
        else:
            raise ValueError(f"Unknown stat value: {value}")
        total += values.sum(dim=0)
        total_sq += (values * values).sum(dim=0)
        count += values.shape[0]
    mean = total / max(count, 1)
    var = (total_sq / max(count, 1) - mean.square()).clamp_min(1e-12)
    std = torch.sqrt(var)
    return {
        "mean": mean.float().tolist(),
        "std": std.float().tolist(),
        "value": value,
        "samples": len(train_index),
        "sampling": sampling,
    }


def compute_velocity_stats(
    train_index: Sequence[tuple[Path, int]],
    cache: CaseCache,
    max_samples: int | None = 128,
    sampling: str = "uniform",
    seed: int = 2026,
) -> dict[str, object]:
    return compute_vector_stats(train_index, cache, "velocity", max_samples, sampling, seed)


def compute_delta_stats(
    train_index: Sequence[tuple[Path, int]],
    cache: CaseCache,
    max_samples: int | None = 128,
    sampling: str = "uniform",
    seed: int = 2026,
) -> dict[str, object]:
    return compute_vector_stats(train_index, cache, "delta", max_samples, sampling, seed)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def make_lr(optimizer: torch.optim.Optimizer, base_lr: float, min_lr: float, epoch: int, decay_start: int, total_epochs: int) -> float:
    if epoch < decay_start:
        lr = base_lr
    else:
        span = max(total_epochs - decay_start, 1)
        progress = min(max((epoch - decay_start + 1) / span, 0.0), 1.0)
        lr = base_lr * ((min_lr / base_lr) ** progress)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr
