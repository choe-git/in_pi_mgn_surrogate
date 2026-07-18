from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from .data import DT_SECONDS, GraphSample, N_NODE_TYPES, one_hot_node_type


AUGMENTATION_IDS = ("1", "2", "3")


@dataclass(frozen=True)
class TrainingAugmentation:
    """Training-only perturbations for every dynamic model input."""

    current_u: torch.Tensor
    prev_u: torch.Tensor
    inflow_context: torch.Tensor


def make_mlp(in_dim: int, out_dim: int, hidden_dim: int = 128, hidden_layers: int = 2) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for _ in range(hidden_layers):
        layers.append(nn.Linear(prev, hidden_dim))
        layers.append(nn.ReLU())
        prev = hidden_dim
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class GraphNetBlock(nn.Module):
    def __init__(self, latent_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.edge_mlp = make_mlp(latent_dim * 3, latent_dim, hidden_dim)
        self.node_mlp = make_mlp(latent_dim * 2, latent_dim, hidden_dim)

    def forward(
        self,
        node_latent: torch.Tensor,
        edge_latent: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index
        edge_input = torch.cat([edge_latent, node_latent[src], node_latent[dst]], dim=-1)
        edge_update = self.edge_mlp(edge_input)
        edge_latent = edge_latent + edge_update

        agg = torch.zeros_like(node_latent)
        agg.index_add_(0, dst, edge_latent)
        node_update = self.node_mlp(torch.cat([node_latent, agg], dim=-1))
        node_latent = node_latent + node_update
        return node_latent, edge_latent


class MeshGraphNet(nn.Module):
    def __init__(
        self,
        node_input_dim: int,
        edge_input_dim: int = 4,
        latent_dim: int = 128,
        hidden_dim: int = 128,
        message_passing_steps: int = 15,
    ):
        super().__init__()
        self.node_encoder = make_mlp(node_input_dim, latent_dim, hidden_dim)
        self.edge_encoder = make_mlp(edge_input_dim, latent_dim, hidden_dim)
        self.processor = nn.ModuleList(
            [GraphNetBlock(latent_dim, hidden_dim) for _ in range(message_passing_steps)]
        )
        self.decoder = make_mlp(latent_dim, 3, hidden_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        node_latent = self.node_encoder(node_features)
        edge_latent = self.edge_encoder(edge_attr)
        for block in self.processor:
            node_latent, edge_latent = block(node_latent, edge_latent, edge_index)
        return self.decoder(node_latent)


def node_feature_dim(model_variant: str) -> int:
    model_variant = model_variant.lower()
    # current velocity, normalized position, normalized time, node type one-hot
    dim = 3 + 3 + 1 + N_NODE_TYPES
    if "in" in model_variant:
        # acceleration feature plus inflow context: mean/min/max inlet speed at t+1
        dim += 3 + 3
    return dim


def _smooth_noise(
    noise: torch.Tensor,
    edge_index: torch.Tensor,
    fixed_mask: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    """Apply graph-neighborhood averaging while preserving fixed boundaries."""

    if steps <= 0:
        return noise

    src, dst = edge_index
    degree = torch.zeros((noise.shape[0], 1), dtype=noise.dtype, device=noise.device)
    degree.index_add_(0, dst, torch.ones((dst.numel(), 1), dtype=noise.dtype, device=noise.device))
    for _ in range(steps):
        neighbor_sum = torch.zeros_like(noise)
        neighbor_sum.index_add_(0, dst, noise[src])
        noise = (noise + neighbor_sum) / (degree + 1.0)
        noise = noise.masked_fill(fixed_mask[:, None], 0.0)
    return noise


def _unit_noise(
    sample: GraphSample,
    fixed_mask: torch.Tensor,
    use_smoothing: bool,
    smoothing_steps: int,
) -> torch.Tensor:
    """Sample dynamic-node noise with unit RMS per velocity component."""

    noise = torch.randn_like(sample.current_u)
    noise = noise.masked_fill(fixed_mask[:, None], 0.0)
    if use_smoothing:
        noise = _smooth_noise(noise, sample.edge_index, fixed_mask, smoothing_steps)

    dynamic_mask = ~fixed_mask
    if dynamic_mask.any():
        rms = noise[dynamic_mask].square().mean(dim=0).sqrt().clamp_min(1e-6)
        noise = noise / rms
    return noise.masked_fill(fixed_mask[:, None], 0.0)


def augment_training_inputs(
    sample: GraphSample,
    velocity_std: torch.Tensor,
    augmentation_ids: Sequence[str],
    noise_std: float,
    temporal_noise_correlation: float,
    spatial_noise_smoothing_steps: int,
    magnitude_jitter_std: float,
) -> TrainingAugmentation:
    """Perturb current/previous velocity and inflow context only during training.

    1: temporally correlated additive velocity noise.
    2: spatially smooth graph noise instead of independent node noise.
    3: one global magnitude factor for the dynamic velocity inputs and inflow.

    Wall and inlet velocity values remain exact. The local acceleration feature
    is derived from the two perturbed velocity states, so it receives the
    corresponding temporally consistent perturbation without an extra branch.
    """

    selected = set(augmentation_ids)
    unknown = selected.difference(AUGMENTATION_IDS)
    if unknown:
        raise ValueError(f"Unknown velocity augmentation ids: {sorted(unknown)}")
    if not 0.0 <= temporal_noise_correlation <= 1.0:
        raise ValueError("temporal_noise_correlation must be in [0, 1]")
    if spatial_noise_smoothing_steps < 0:
        raise ValueError("spatial_noise_smoothing_steps must be non-negative")
    if magnitude_jitter_std < 0.0:
        raise ValueError("magnitude_jitter_std must be non-negative")

    current_u = sample.current_u
    prev_u = sample.prev_u
    inflow_context = sample.inflow_context
    fixed_mask = sample.wall_mask | sample.inlet_mask

    use_additive_noise = noise_std > 0.0 and bool(selected.intersection({"1", "2"}))
    if use_additive_noise:
        use_smoothing = "2" in selected
        if "1" in selected:
            shared = _unit_noise(sample, fixed_mask, use_smoothing, spatial_noise_smoothing_steps)
            current_innovation = _unit_noise(sample, fixed_mask, use_smoothing, spatial_noise_smoothing_steps)
            prev_innovation = _unit_noise(sample, fixed_mask, use_smoothing, spatial_noise_smoothing_steps)
            shared_weight = temporal_noise_correlation**0.5
            innovation_weight = (1.0 - temporal_noise_correlation) ** 0.5
            current_noise = shared_weight * shared + innovation_weight * current_innovation
            prev_noise = shared_weight * shared + innovation_weight * prev_innovation
        else:
            current_noise = _unit_noise(sample, fixed_mask, use_smoothing, spatial_noise_smoothing_steps)
            prev_noise = _unit_noise(sample, fixed_mask, use_smoothing, spatial_noise_smoothing_steps)

        current_u = current_u + noise_std * current_noise * velocity_std
        prev_u = prev_u + noise_std * prev_noise * velocity_std

        # Inflow is a global three-value feature, so use one shared perturbation
        # for all nodes rather than independent per-node values.
        context_noise = torch.randn(
            (1, 3), dtype=inflow_context.dtype, device=inflow_context.device
        ) * (noise_std * velocity_std.norm())
        inflow_context = (inflow_context + context_noise).clamp_min(0.0)

    if "3" in selected and magnitude_jitter_std > 0.0:
        magnitude_factor = 1.0 + torch.randn((), dtype=current_u.dtype, device=current_u.device) * magnitude_jitter_std
        magnitude_factor = magnitude_factor.clamp_min(0.25)
        dynamic_mask = (~fixed_mask)[:, None]
        current_u = torch.where(dynamic_mask, current_u * magnitude_factor, sample.current_u)
        prev_u = torch.where(dynamic_mask, prev_u * magnitude_factor, sample.prev_u)
        inflow_context = (inflow_context * magnitude_factor).clamp_min(0.0)

    return TrainingAugmentation(
        current_u=current_u,
        prev_u=prev_u,
        inflow_context=inflow_context,
    )


def build_node_features(
    sample: GraphSample,
    model_variant: str,
    velocity_mean: torch.Tensor,
    velocity_std: torch.Tensor,
    current_u: torch.Tensor | None = None,
    prev_u: torch.Tensor | None = None,
    inflow_context: torch.Tensor | None = None,
    acceleration_mode: str = "acceleration",
) -> torch.Tensor:
    if current_u is None:
        current_u = sample.current_u
    if prev_u is None:
        prev_u = sample.prev_u
    if inflow_context is None:
        inflow_context = sample.inflow_context

    u_norm = (current_u - velocity_mean) / velocity_std
    point_center = sample.points.mean(dim=0, keepdim=True)
    point_scale = sample.points.std().clamp_min(1e-6)
    point_norm = (sample.points - point_center) / point_scale
    pieces = [u_norm, point_norm, sample.time_feature, one_hot_node_type(sample.node_type)]
    if "in" in model_variant.lower():
        if acceleration_mode == "physical_acceleration":
            acceleration = (current_u - prev_u) / DT_SECONDS
        elif acceleration_mode == "acceleration":
            acceleration = current_u - prev_u
        else:
            raise ValueError(f"Unknown acceleration_mode: {acceleration_mode}")
        acceleration_norm = acceleration / velocity_std
        inflow_norm = inflow_context / velocity_std.norm().clamp_min(1e-6)
        pieces.extend([acceleration_norm, inflow_norm])
    return torch.cat(pieces, dim=-1)
