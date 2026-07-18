from __future__ import annotations

import torch
from torch import nn

from .data import DT_SECONDS, GraphSample, N_NODE_TYPES, one_hot_node_type


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
        # acceleration plus inflow context: mean/min/max inlet speed at t+1
        dim += 3 + 3
    return dim


def add_training_noise(
    sample: GraphSample,
    velocity_std: torch.Tensor,
    noise_std: float,
) -> torch.Tensor:
    if noise_std <= 0:
        return sample.current_u
    fixed_mask = sample.wall_mask | sample.inlet_mask
    noise = torch.randn_like(sample.current_u) * noise_std * velocity_std
    noise = noise.masked_fill(fixed_mask[:, None], 0.0)
    return sample.current_u + noise


def build_node_features(
    sample: GraphSample,
    model_variant: str,
    velocity_mean: torch.Tensor,
    velocity_std: torch.Tensor,
    current_u: torch.Tensor | None = None,
    prev_u: torch.Tensor | None = None,
    noise_std: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    if current_u is None:
        current_u = add_training_noise(sample, velocity_std, noise_std) if training else sample.current_u
    if prev_u is None:
        prev_u = sample.prev_u

    u_norm = (current_u - velocity_mean) / velocity_std
    point_center = sample.points.mean(dim=0, keepdim=True)
    point_scale = sample.points.std().clamp_min(1e-6)
    point_norm = (sample.points - point_center) / point_scale
    pieces = [u_norm, point_norm, sample.time_feature, one_hot_node_type(sample.node_type)]
    if "in" in model_variant.lower():
        accel = (current_u - prev_u) / DT_SECONDS
        accel_norm = accel / velocity_std
        inflow_norm = sample.inflow_context / velocity_std.norm().clamp_min(1e-6)
        pieces.extend([accel_norm, inflow_norm])
    return torch.cat(pieces, dim=-1)
