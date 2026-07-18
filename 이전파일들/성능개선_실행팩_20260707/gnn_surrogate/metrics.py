from __future__ import annotations

import torch


def rmse_mm_s(pred_u: torch.Tensor, true_u: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    err = pred_u[mask] - true_u[mask]
    return torch.sqrt(torch.mean(torch.sum(err.square(), dim=-1)))


def mean_and_sd(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    t = torch.tensor(values, dtype=torch.float32)
    return float(t.mean()), float(t.std(unbiased=False))
