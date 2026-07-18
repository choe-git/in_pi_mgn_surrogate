from __future__ import annotations

import torch


def _edge_unit(points: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    src, dst = edge_index
    rel = points[dst] - points[src]
    dist = torch.linalg.norm(rel, dim=-1, keepdim=True).clamp_min(1e-8)
    return rel / dist, dist


def node_divergence_like(
    velocity: torch.Tensor,
    points: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    src, dst = edge_index
    unit, dist = _edge_unit(points, edge_index)
    edge_flux = ((velocity[dst] - velocity[src]) * unit).sum(dim=-1, keepdim=True) / dist
    div = torch.zeros((velocity.shape[0], 1), dtype=velocity.dtype, device=velocity.device)
    div.index_add_(0, src, edge_flux)
    return div


def graph_laplacian(
    velocity: torch.Tensor,
    points: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    src, dst = edge_index
    _, dist = _edge_unit(points, edge_index)
    diff = (velocity[dst] - velocity[src]) / dist.square()
    lap = torch.zeros_like(velocity)
    deg = torch.zeros((velocity.shape[0], 1), dtype=velocity.dtype, device=velocity.device)
    lap.index_add_(0, src, diff)
    deg.index_add_(0, src, torch.ones_like(dist))
    return lap / deg.clamp_min(1.0)


def node_gradient_like(
    velocity: torch.Tensor,
    points: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    src, dst = edge_index
    rel = points[dst] - points[src]
    dist_sq = torch.sum(rel.square(), dim=-1, keepdim=True).clamp_min(1e-8)
    edge_grad = (velocity[dst] - velocity[src]).unsqueeze(-1) * rel.unsqueeze(1) / dist_sq.unsqueeze(-1)
    grad = torch.zeros(
        (velocity.shape[0], velocity.shape[1], points.shape[1]),
        dtype=velocity.dtype,
        device=velocity.device,
    )
    deg = torch.zeros((velocity.shape[0], 1, 1), dtype=velocity.dtype, device=velocity.device)
    grad.index_add_(0, src, edge_grad)
    deg.index_add_(0, src, torch.ones((edge_index.shape[1], 1, 1), dtype=velocity.dtype, device=velocity.device))
    return grad / deg.clamp_min(1.0)


def convective_term(velocity: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    return torch.einsum("nb,nab->na", velocity, grad)


def physics_losses(
    pred_u: torch.Tensor,
    true_u: torch.Tensor,
    current_u: torch.Tensor,
    points: torch.Tensor,
    edge_index: torch.Tensor,
    mask: torch.Tensor,
    operator_mode: str = "legacy",
    continuity_target: str = "zero",
) -> dict[str, torch.Tensor]:
    """Physics-inspired losses using local graph finite differences.

    The paper includes continuity, convection, and viscosity terms. The released
    dataset does not expose all solver-side quantities, so this implementation
    matches predicted and true local operators on the graph neighborhood.
    """

    if operator_mode == "legacy":
        mask1 = mask[:, None]
        pred_div = node_divergence_like(pred_u, points, edge_index)
        true_div = node_divergence_like(true_u, points, edge_index)
        if continuity_target == "zero":
            continuity = torch.mean(pred_div[mask1].square())
        elif continuity_target == "match":
            continuity = torch.mean((pred_div[mask1] - true_div[mask1]).square())
        else:
            raise ValueError(f"Unknown continuity_target: {continuity_target}")

        pred_lap = graph_laplacian(pred_u, points, edge_index)
        true_lap = graph_laplacian(true_u, points, edge_index)
        viscosity = torch.mean((pred_lap[mask] - true_lap[mask]).square())

        pred_adv = pred_u * graph_laplacian(current_u, points, edge_index)
        true_adv = true_u * graph_laplacian(current_u, points, edge_index)
        convection = torch.mean((pred_adv[mask] - true_adv[mask]).square())
    elif operator_mode == "gradient":
        pred_grad = node_gradient_like(pred_u, points, edge_index)
        true_grad = node_gradient_like(true_u, points, edge_index)
        pred_div = torch.diagonal(pred_grad, dim1=-2, dim2=-1).sum(dim=-1)
        true_div = torch.diagonal(true_grad, dim1=-2, dim2=-1).sum(dim=-1)
        if continuity_target == "zero":
            continuity = torch.mean(pred_div[mask].square())
        elif continuity_target == "match":
            continuity = torch.mean((pred_div[mask] - true_div[mask]).square())
        else:
            raise ValueError(f"Unknown continuity_target: {continuity_target}")

        pred_lap = graph_laplacian(pred_u, points, edge_index)
        true_lap = graph_laplacian(true_u, points, edge_index)
        viscosity = torch.mean((pred_lap[mask] - true_lap[mask]).square())

        pred_adv = convective_term(pred_u, pred_grad)
        true_adv = convective_term(true_u, true_grad)
        convection = torch.mean((pred_adv[mask] - true_adv[mask]).square())
    else:
        raise ValueError(f"Unknown physics operator mode: {operator_mode}")

    return {
        "continuity": continuity,
        "convection": convection,
        "viscosity": viscosity,
    }
