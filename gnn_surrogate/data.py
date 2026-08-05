from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import torch
import h5py
import numpy as np


VELOCITY_START_KEY = 2
VELOCITY_STRIDE = 2
N_TIMESTEPS = 80
DT_SECONDS = 0.01

NODE_INTERIOR = 0
NODE_WALL = 1
NODE_INLET = 2
NODE_OUTLET = 3
N_NODE_TYPES = 4


@dataclass(frozen=True)
class GraphMetadata:
    case_name: str
    path: Path
    n_nodes: int
    n_cells: int
    n_edges: int


@dataclass
class GraphSample:
    case_name: str
    time_index: int
    points: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    node_type: torch.Tensor
    wall_mask: torch.Tensor
    inlet_mask: torch.Tensor
    outlet_mask: torch.Tensor
    current_u: torch.Tensor
    prev_u: torch.Tensor
    target_u: torch.Tensor
    target_acceleration: torch.Tensor
    time_feature: torch.Tensor
    inflow_context: torch.Tensor


def resolve_data_dir(data_dir: str | Path) -> Path:
    data_path = Path(data_dir).expanduser().resolve()
    if (data_path / "coarse_dataset").is_dir():
        data_path = data_path / "coarse_dataset"
    if not data_path.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_path}")
    return data_path


def list_h5_files(data_dir: str | Path) -> list[Path]:
    data_path = resolve_data_dir(data_dir)
    files = sorted(data_path.glob("AllFields_Resultats_MESH_*.h5"))
    if not files:
        files = sorted(data_path.glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No .h5 files found under {data_path}")
    return files


def case_id(path: Path) -> str:
    return path.stem.replace("AllFields_Resultats_MESH_", "")


def velocity_key(time_index: int) -> str:
    return f"data{VELOCITY_START_KEY + VELOCITY_STRIDE * time_index}"


def wall_key(time_index: int) -> str:
    return f"data{VELOCITY_START_KEY + VELOCITY_STRIDE * time_index + 1}"


def read_static_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h5:
        points = h5["data0"][...].astype(np.float32)
        cells = h5["data1"][...].astype(np.int64)
    return points, cells


def read_velocity(path: Path, time_index: int) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        return h5[velocity_key(time_index)][...].astype(np.float32)


def read_wall_mask(path: Path, time_index: int = 0) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        return h5[wall_key(time_index)][...].astype(np.float32) > 0.5


def tetrahedra_to_edges(cells: np.ndarray) -> np.ndarray:
    pairs = np.array(
        [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=np.int64
    )
    undirected = cells[:, pairs].reshape(-1, 2)
    directed = np.concatenate([undirected, undirected[:, ::-1]], axis=0)
    directed = np.unique(directed, axis=0)
    directed = directed[directed[:, 0] != directed[:, 1]]
    return directed.T.astype(np.int64)


def tetrahedra_boundary_faces(cells: np.ndarray) -> np.ndarray:
    faces = np.concatenate(
        [
            cells[:, [0, 1, 2]],
            cells[:, [0, 1, 3]],
            cells[:, [0, 2, 3]],
            cells[:, [1, 2, 3]],
        ],
        axis=0,
    )
    faces = np.sort(faces, axis=1)
    unique_faces, counts = np.unique(faces, axis=0, return_counts=True)
    return unique_faces[counts == 1].astype(np.int64)


def _oriented_boundary_faces(
    points: np.ndarray,
    cells: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return boundary faces with outward unit normals and triangle areas."""
    face_patterns = np.array(
        [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
        dtype=np.int64,
    )
    opposite_patterns = np.array([3, 2, 1, 0], dtype=np.int64)
    raw_faces = cells[:, face_patterns].reshape(-1, 3)
    raw_opposite_nodes = cells[:, opposite_patterns].reshape(-1)

    face_keys = np.sort(raw_faces, axis=1)
    _, first_indices, counts = np.unique(
        face_keys,
        axis=0,
        return_index=True,
        return_counts=True,
    )
    boundary_indices = first_indices[counts == 1]
    faces = raw_faces[boundary_indices].copy()
    opposite_nodes = raw_opposite_nodes[boundary_indices]
    if faces.size == 0:
        return (
            np.empty((0, 3), dtype=np.int64),
            np.empty((0, 3), dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    x0 = points[faces[:, 0]].astype(np.float64, copy=False)
    x1 = points[faces[:, 1]].astype(np.float64, copy=False)
    x2 = points[faces[:, 2]].astype(np.float64, copy=False)
    centroids = (x0 + x1 + x2) / 3.0
    raw_normals = np.cross(x1 - x0, x2 - x0)
    normal_lengths = np.linalg.norm(raw_normals, axis=1)
    valid = normal_lengths > np.finfo(np.float64).eps
    faces = faces[valid]
    opposite_nodes = opposite_nodes[valid]
    centroids = centroids[valid]
    raw_normals = raw_normals[valid]
    normal_lengths = normal_lengths[valid]
    if faces.size == 0:
        return (
            np.empty((0, 3), dtype=np.int64),
            np.empty((0, 3), dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    points_to_interior = points[opposite_nodes] - centroids
    inward = np.einsum("ij,ij->i", raw_normals, points_to_interior) > 0.0
    raw_normals[inward] *= -1.0

    normals = raw_normals / normal_lengths[:, None]
    areas = 0.5 * normal_lengths
    return faces.astype(np.int64, copy=False), normals, areas


def _connected_face_components(faces: np.ndarray) -> list[np.ndarray]:
    """Group triangle faces that share complete edges."""
    n_faces = int(faces.shape[0])
    parent = np.arange(n_faces, dtype=np.int64)
    rank = np.zeros(n_faces, dtype=np.int8)

    def find(index: int) -> int:
        root = index
        while parent[root] != root:
            root = int(parent[root])
        while parent[index] != index:
            next_index = int(parent[index])
            parent[index] = root
            index = next_index
        return root

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if rank[left_root] < rank[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        if rank[left_root] == rank[right_root]:
            rank[left_root] += 1

    edge_owner: dict[tuple[int, int], int] = {}
    for face_index, face in enumerate(faces):
        edges = ((face[0], face[1]), (face[1], face[2]), (face[2], face[0]))
        for left, right in edges:
            edge = (int(min(left, right)), int(max(left, right)))
            owner = edge_owner.get(edge)
            if owner is None:
                edge_owner[edge] = face_index
            else:
                union(face_index, owner)

    groups: dict[int, list[int]] = {}
    for face_index in range(n_faces):
        groups.setdefault(find(face_index), []).append(face_index)
    return [np.asarray(group, dtype=np.int64) for group in groups.values()]


def _open_cap_components(
    faces: np.ndarray,
    areas: np.ndarray,
    wall_mask: np.ndarray,
) -> list[np.ndarray]:
    """Find the two largest connected non-wall boundary-face patches."""
    if faces.size == 0:
        return []
    candidate_indices = np.flatnonzero(np.any(~wall_mask[faces], axis=1))
    if candidate_indices.size == 0:
        return []

    local_components = _connected_face_components(faces[candidate_indices])
    components = [candidate_indices[component] for component in local_components]
    components.sort(
        key=lambda component: float(areas[component].sum()),
        reverse=True,
    )
    return components[:2] if len(components) >= 2 else []


def _cap_flux(
    velocities: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
) -> np.ndarray:
    """Compute area-weighted signed flow through one boundary cap."""
    mean_face_velocity = velocities[:, faces].mean(axis=2)
    area_vectors = normals * areas[:, None]
    return np.einsum("tfj,fj->t", mean_face_velocity, area_vectors)


def _flux_sign_persistence(flux: np.ndarray, expected_sign: float) -> float:
    peak = float(np.max(np.abs(flux))) if flux.size else 0.0
    if peak <= np.finfo(np.float64).eps:
        return 0.0
    active = np.abs(flux) >= 0.05 * peak
    if not np.any(active):
        return 0.0
    return float(np.mean(expected_sign * flux[active] > 0.0))


def infer_zero_velocity_wall_mask(path: Path, atol: float = 1e-8) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        wall_mask = None
        for t in range(N_TIMESTEPS):
            velocity = h5[velocity_key(t)][...].astype(np.float32)
            is_zero = np.linalg.norm(velocity, axis=1) <= atol
            wall_mask = is_zero if wall_mask is None else (wall_mask & is_zero)
    return wall_mask


def _principal_axis(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0].astype(np.float32)
    norm = np.linalg.norm(axis)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return axis / norm


def _split_open_caps(
    points: np.ndarray,
    open_boundary_nodes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cap_points = points[open_boundary_nodes]
    axis = _principal_axis(cap_points)
    coords = cap_points @ axis
    centers = np.array([coords.min(), coords.max()], dtype=np.float32)
    labels = None

    for _ in range(16):
        new_labels = np.argmin(np.abs(coords[:, None] - centers[None, :]), axis=1)
        if labels is not None and np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for group in (0, 1):
            if np.any(labels == group):
                centers[group] = coords[labels == group].mean()

    if labels is None:
        labels = np.zeros(coords.shape[0], dtype=np.int64)

    low_group = int(np.argmin(centers))
    high_group = 1 - low_group
    return open_boundary_nodes[labels == low_group], open_boundary_nodes[labels == high_group], axis


def _mean_velocity(path: Path, nodes: np.ndarray) -> np.ndarray:
    if nodes.size == 0:
        return np.zeros(3, dtype=np.float32)
    total = np.zeros(3, dtype=np.float64)
    count = 0
    with h5py.File(path, "r") as h5:
        for t in range(N_TIMESTEPS):
            velocity = h5[velocity_key(t)][...].astype(np.float32)
            total += velocity[nodes].mean(axis=0)
            count += 1
    return (total / max(count, 1)).astype(np.float32)


def _infer_open_boundary_masks_geometric(
    points: np.ndarray,
    cells: np.ndarray,
    wall_mask: np.ndarray,
    path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    boundary_faces = tetrahedra_boundary_faces(cells)
    boundary_mask = np.zeros(points.shape[0], dtype=bool)
    boundary_mask[np.unique(boundary_faces.reshape(-1))] = True

    open_boundary_nodes = np.flatnonzero(boundary_mask & (~wall_mask))
    if open_boundary_nodes.size < 4:
        return np.zeros(points.shape[0], dtype=bool), np.zeros(points.shape[0], dtype=bool)

    low_cap, high_cap, axis = _split_open_caps(points, open_boundary_nodes)
    if low_cap.size == 0 or high_cap.size == 0:
        return np.zeros(points.shape[0], dtype=bool), np.zeros(points.shape[0], dtype=bool)

    low_velocity = _mean_velocity(path, low_cap)
    high_velocity = _mean_velocity(path, high_cap)

    low_inflow_score = float(np.dot(low_velocity, axis))
    high_inflow_score = float(np.dot(high_velocity, -axis))
    if high_inflow_score > low_inflow_score:
        inlet_nodes, outlet_nodes = high_cap, low_cap
    else:
        inlet_nodes, outlet_nodes = low_cap, high_cap

    inlet_mask = np.zeros(points.shape[0], dtype=bool)
    outlet_mask = np.zeros(points.shape[0], dtype=bool)
    inlet_mask[inlet_nodes] = True
    outlet_mask[outlet_nodes] = True
    return inlet_mask, outlet_mask


def infer_open_boundary_masks(
    points: np.ndarray,
    cells: np.ndarray,
    wall_mask: np.ndarray,
    path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Infer inlet/outlet from outward-oriented, area-weighted cap flux."""
    faces, normals, areas = _oriented_boundary_faces(points, cells)
    components = _open_cap_components(faces, areas, wall_mask)
    if len(components) != 2:
        return _infer_open_boundary_masks_geometric(points, cells, wall_mask, path)

    with h5py.File(path, "r") as h5:
        velocities = np.stack(
            [h5[velocity_key(t)][...].astype(np.float32) for t in range(N_TIMESTEPS)],
            axis=0,
        )

    fluxes = [
        _cap_flux(
            velocities,
            faces[component],
            normals[component],
            areas[component],
        )
        for component in components
    ]
    mean_fluxes = np.asarray([np.mean(flux) for flux in fluxes], dtype=np.float64)
    flux_scale = float(np.max(np.abs(mean_fluxes)))
    tolerance = max(np.finfo(np.float64).eps, flux_scale * 1e-6)
    inlet_candidates = np.flatnonzero(mean_fluxes < -tolerance)
    outlet_candidates = np.flatnonzero(mean_fluxes > tolerance)
    if inlet_candidates.size != 1 or outlet_candidates.size != 1:
        return _infer_open_boundary_masks_geometric(points, cells, wall_mask, path)

    inlet_cap = int(inlet_candidates[0])
    outlet_cap = int(outlet_candidates[0])
    if (
        _flux_sign_persistence(fluxes[inlet_cap], -1.0) < 0.8
        or _flux_sign_persistence(fluxes[outlet_cap], 1.0) < 0.8
    ):
        return _infer_open_boundary_masks_geometric(points, cells, wall_mask, path)

    absolute_flux = np.abs(fluxes[inlet_cap]) + np.abs(fluxes[outlet_cap])
    active = absolute_flux >= max(
        float(np.max(absolute_flux)) * 0.01,
        np.finfo(np.float64).eps,
    )
    mass_residual = (
        np.abs(fluxes[inlet_cap] + fluxes[outlet_cap])[active]
        / absolute_flux[active]
    )
    if mass_residual.size == 0 or float(np.mean(mass_residual)) > 0.2:
        return _infer_open_boundary_masks_geometric(points, cells, wall_mask, path)

    inlet_nodes = np.unique(faces[components[inlet_cap]].reshape(-1))
    outlet_nodes = np.unique(faces[components[outlet_cap]].reshape(-1))
    inlet_mask = np.zeros(points.shape[0], dtype=bool)
    outlet_mask = np.zeros(points.shape[0], dtype=bool)
    inlet_mask[inlet_nodes] = True
    outlet_mask[outlet_nodes] = True
    inlet_mask &= ~wall_mask
    outlet_mask &= ~wall_mask
    outlet_mask &= ~inlet_mask
    if not inlet_mask.any() or not outlet_mask.any():
        return _infer_open_boundary_masks_geometric(points, cells, wall_mask, path)
    return inlet_mask, outlet_mask


def find_nonzero_derivate(
    data,
    diff_stride=1,
    score_threshold=None,
    small_diff_threshold=1e-12,
):
    velocities = np.stack(data[2::2], axis=0)  # [T, N, 3]
    masks = np.stack(data[3::2], axis=0)       # [T, N] or [T, N, 1]
    if masks.ndim == 3 and masks.shape[-1] == 1:
        masks = masks[..., 0]
    masks = masks.astype(bool)  # [T, N]
    mask_any = masks.any(axis=0)  # [N]
    true_zero_wall_mask = np.all(velocities == 0, axis=(0, 2))  # [N]
    candidate_base_mask = mask_any & (~true_zero_wall_mask)  # [N]
    diff = velocities[diff_stride:] - velocities[:-diff_stride]
    diff_norm = np.linalg.norm(diff, axis=-1)  # [T-diff_stride, N]
    mean_diff_norm = diff_norm.mean(axis=0)  # [N]
    std_diff_norm = diff_norm.std(axis=0)    # [N]
    candidate_mask = candidate_base_mask.copy()
    if score_threshold is not None:
        candidate_mask = candidate_mask & (mean_diff_norm > score_threshold)
    candidate_indices = np.where(candidate_mask)[0]
    small_mean_diff_mask = mean_diff_norm < small_diff_threshold  # [N]
    small_mean_diff_candidate_mask = candidate_mask & small_mean_diff_mask  # [N]
    small_mean_diff_indices = np.where(small_mean_diff_mask)[0]
    small_mean_diff_candidate_indices = np.where(small_mean_diff_candidate_mask)[0]

    return {
        "true_zero_wall_mask": true_zero_wall_mask,
        "candidate_base_mask": candidate_base_mask,
        "small_mean_diff_candidate_mask": small_mean_diff_candidate_mask,
    }

def infer_node_types(
    points: np.ndarray,
    cells: np.ndarray,
    wall_mask: np.ndarray,
    boundary_percentile: float = 2.0,
    path=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Infer wall/inlet/outlet node types.

    The released files expose a boundary-like wall_mask but no explicit
    inlet/outlet labels. We first recover no-slip wall nodes from zero velocity
    over the whole trajectory, then split non-wall boundary caps geometrically.
    """

    if path is None:
        true_zero_wall_mask = wall_mask.astype(bool)
        inlet_mask = np.zeros(points.shape[0], dtype=bool)
        outlet_mask = np.zeros(points.shape[0], dtype=bool)
    else:
        path = Path(path)
        true_zero_wall_mask = infer_zero_velocity_wall_mask(path)
        inlet_mask, outlet_mask = infer_open_boundary_masks(
            points,
            cells,
            true_zero_wall_mask,
            path,
        )
        if not inlet_mask.any() or not outlet_mask.any():
            with h5py.File(path, "r") as f:
                data = [f[f"data{i}"][:] for i in range(162)]
            diff_dict = find_nonzero_derivate(data)
            inlet_mask = diff_dict["small_mean_diff_candidate_mask"]
            outlet_mask = diff_dict["candidate_base_mask"] & (~inlet_mask)

    inlet_mask = inlet_mask & (~true_zero_wall_mask)
    outlet_mask = outlet_mask & (~true_zero_wall_mask) & (~inlet_mask)

    node_type = np.full(points.shape[0], NODE_INTERIOR, dtype=np.int64)
    node_type[true_zero_wall_mask] = NODE_WALL
    node_type[inlet_mask] = NODE_INLET
    node_type[outlet_mask] = NODE_OUTLET

    return node_type, inlet_mask, outlet_mask


def build_edge_attr(points: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    src, dst = edge_index
    rel = points[dst] - points[src]
    dist = np.linalg.norm(rel, axis=1, keepdims=True)
    scale = float(np.std(points))
    if not math.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    return np.concatenate([rel / scale, dist / scale], axis=1).astype(np.float32)


def one_hot_node_type(node_type: torch.Tensor) -> torch.Tensor:
    import torch

    return torch.nn.functional.one_hot(node_type.long(), N_NODE_TYPES).float()


class CaseCache:
    def __init__(self, boundary_percentile: float = 2.0):
        self.boundary_percentile = boundary_percentile
        self._cache: dict[Path, dict[str, np.ndarray]] = {}
        self._velocity_cache: dict[Path, np.ndarray] = {}
        self._tensor_cache: dict[tuple[Path, str], dict[str, object]] = {}

    def static(self, path: Path) -> dict[str, np.ndarray]:
        path = path.resolve()
        cached = self._cache.get(path)
        if cached is not None:
            return cached
        points, cells = read_static_mesh(path)
        raw_wall_mask = read_wall_mask(path, 0)
        edge_index = tetrahedra_to_edges(cells)
        edge_attr = build_edge_attr(points, edge_index)
        node_type, inlet_mask, outlet_mask = infer_node_types(
            points, cells, raw_wall_mask, self.boundary_percentile, path
        )
        wall_mask = node_type == NODE_WALL
        cached = {
            "points": points,
            "cells": cells,
            "wall_mask": wall_mask,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "node_type": node_type,
            "inlet_mask": inlet_mask,
            "outlet_mask": outlet_mask,
        }
        self._cache[path] = cached
        return cached

    def velocities(self, path: Path) -> np.ndarray:
        path = path.resolve()
        cached = self._velocity_cache.get(path)
        if cached is not None:
            return cached
        with h5py.File(path, "r") as h5:
            cached = np.stack(
                [h5[velocity_key(t)][...].astype(np.float32) for t in range(N_TIMESTEPS)],
                axis=0,
            )
        self._velocity_cache[path] = cached
        return cached

    def velocity(self, path: Path, time_index: int) -> np.ndarray:
        return self.velocities(path)[time_index]

    def tensors(self, path: Path, device: torch.device | str = "cpu") -> dict[str, object]:
        import torch

        path = path.resolve()
        device_key = str(torch.device(device))
        key = (path, device_key)
        cached = self._tensor_cache.get(key)
        if cached is not None:
            return cached

        static = self.static(path)
        cached = {
            "points": torch.as_tensor(static["points"], dtype=torch.float32, device=device),
            "edge_index": torch.as_tensor(static["edge_index"], dtype=torch.long, device=device),
            "edge_attr": torch.as_tensor(static["edge_attr"], dtype=torch.float32, device=device),
            "node_type": torch.as_tensor(static["node_type"], dtype=torch.long, device=device),
            "wall_mask": torch.as_tensor(static["wall_mask"], dtype=torch.bool, device=device),
            "inlet_mask": torch.as_tensor(static["inlet_mask"], dtype=torch.bool, device=device),
            "outlet_mask": torch.as_tensor(static["outlet_mask"], dtype=torch.bool, device=device),
        }
        self._tensor_cache[key] = cached
        return cached


def apply_velocity_boundary(
    velocity: torch.Tensor,
    sample: GraphSample,
    clamp_inlet: bool = True,
) -> torch.Tensor:
    velocity = velocity.clone()
    velocity = velocity.masked_fill(sample.wall_mask[:, None], 0.0)
    if clamp_inlet and sample.inlet_mask.any():
        velocity[sample.inlet_mask] = sample.target_u[sample.inlet_mask]
    return velocity


def learned_node_mask(sample: GraphSample) -> torch.Tensor:
    return ~sample.wall_mask


def fluid_node_mask(sample: GraphSample) -> torch.Tensor:
    return ~sample.wall_mask


def inflow_context(target_u: np.ndarray, inlet_mask: np.ndarray) -> np.ndarray:
    if not inlet_mask.any():
        return np.zeros(3, dtype=np.float32)
    mag = np.linalg.norm(target_u[inlet_mask], axis=1)
    return np.array([mag.mean(), mag.min(), mag.max()], dtype=np.float32)


def load_graph_sample(
    path: Path,
    time_index: int,
    cache: CaseCache,
    device: torch.device | str = "cpu",
) -> GraphSample:
    import torch

    if time_index < 1 or time_index >= N_TIMESTEPS - 1:
        raise ValueError("time_index must be in [1, 78] for acceleration features")

    static = cache.static(path)
    static_tensors = cache.tensors(path, device=device)
    prev_u_np = cache.velocity(path, time_index - 1)
    current_u_np = cache.velocity(path, time_index)
    target_u_np = cache.velocity(path, time_index + 1)
    context = inflow_context(target_u_np, static["inlet_mask"])
    n = current_u_np.shape[0]

    prev_u = torch.as_tensor(prev_u_np, dtype=torch.float32, device=device)
    current_u = torch.as_tensor(current_u_np, dtype=torch.float32, device=device)
    target_u = torch.as_tensor(target_u_np, dtype=torch.float32, device=device)

    return GraphSample(
        case_name=case_id(path),
        time_index=time_index,
        points=static_tensors["points"],
        edge_index=static_tensors["edge_index"],
        edge_attr=static_tensors["edge_attr"],
        node_type=static_tensors["node_type"],
        wall_mask=static_tensors["wall_mask"],
        inlet_mask=static_tensors["inlet_mask"],
        outlet_mask=static_tensors["outlet_mask"],
        current_u=current_u,
        prev_u=prev_u,
        target_u=target_u,
        target_acceleration=target_u - current_u,
        time_feature=torch.full((n, 1), time_index / (N_TIMESTEPS - 1), dtype=torch.float32, device=device),
        inflow_context=torch.as_tensor(context, dtype=torch.float32, device=device).expand(n, 3),
    )


def split_files(
    files: Sequence[Path],
    seed: int = 2026,
    test_count: int = 10,
    test_files: Sequence[str] | None = None,
) -> tuple[list[Path], list[Path]]:
    by_name = {p.name: p for p in files}
    by_case = {case_id(p): p for p in files}
    if test_files:
        selected = []
        for item in test_files:
            key = item.strip()
            if not key:
                continue
            if key in by_name:
                selected.append(by_name[key])
            elif key in by_case:
                selected.append(by_case[key])
            else:
                raise KeyError(f"Unknown test file/case id: {key}")
        test = sorted(set(selected))
    else:
        rng = np.random.default_rng(seed)
        idx = sorted(rng.choice(len(files), size=test_count, replace=False).tolist())
        test = [files[i] for i in idx]
    test_set = {p.resolve() for p in test}
    train = [p for p in files if p.resolve() not in test_set]
    return train, test


def resolve_named_files(
    files: Sequence[Path],
    selected_files: Sequence[str],
    split_name: str,
) -> list[Path]:
    by_name = {p.name: p for p in files}
    by_case = {case_id(p): p for p in files}
    selected = []
    for item in selected_files:
        key = item.strip()
        if not key:
            continue
        if key in by_name:
            selected.append(by_name[key])
        elif key in by_case:
            selected.append(by_case[key])
        else:
            raise KeyError(f"Unknown {split_name} file/case id: {key}")
    return sorted(set(selected))


def split_files_three_way(
    files: Sequence[Path],
    seed: int = 2026,
    val_count: int = 5,
    test_count: int = 5,
    val_files: Sequence[str] | None = None,
    test_files: Sequence[str] | None = None,
    split_csv: str | Path | None = None,
) -> tuple[list[Path], list[Path], list[Path]]:
    files = list(files)
    if split_csv is not None:
        csv_path = Path(split_csv).expanduser().resolve()
        if not csv_path.is_file():
            raise FileNotFoundError(f"Split CSV does not exist: {csv_path}")

        by_name = {path.name: path for path in files}
        by_case = {case_id(path): path for path in files}
        assignments: dict[Path, str] = {}
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required_columns = {"data_name", "train", "val", "test"}
            columns = set(reader.fieldnames or [])
            missing_columns = sorted(required_columns - columns)
            if missing_columns:
                raise ValueError(f"Split CSV is missing columns: {missing_columns}")

            for row_number, row in enumerate(reader, start=2):
                data_name = (row.get("data_name") or "").strip()
                key = Path(data_name).name
                path = by_name.get(key) or by_case.get(key)
                if path is None:
                    raise KeyError(f"Unknown data_name at CSV row {row_number}: {data_name}")

                resolved = path.resolve()
                if resolved in assignments:
                    raise ValueError(f"Duplicate data_name at CSV row {row_number}: {data_name}")

                flags = {
                    name: (row.get(name) or "").strip()
                    for name in ("train", "val", "test")
                }
                if any(value not in {"0", "1"} for value in flags.values()):
                    raise ValueError(
                        f"CSV row {row_number} split flags must be binary 0 or 1: {flags}"
                    )
                selected = [name for name, value in flags.items() if value == "1"]
                if len(selected) != 1:
                    raise ValueError(
                        f"CSV row {row_number} must select exactly one of train,val,test: {flags}"
                    )
                assignments[resolved] = selected[0]

        missing_files = sorted(path.name for path in files if path.resolve() not in assignments)
        if missing_files:
            raise ValueError(f"Split CSV does not assign every loaded file: {missing_files}")

        split = {
            name: sorted(path for path in files if assignments[path.resolve()] == name)
            for name in ("train", "val", "test")
        }
        if any(not split[name] for name in split):
            counts = {name: len(paths) for name, paths in split.items()}
            raise ValueError(f"CSV train/val/test splits must all be non-empty: {counts}")
        return split["train"], split["val"], split["test"]

    rng = np.random.default_rng(seed)

    val = resolve_named_files(files, val_files, "validation") if val_files else []
    test = resolve_named_files(files, test_files, "test") if test_files else []
    val_set = {p.resolve() for p in val}
    test_set = {p.resolve() for p in test}
    overlap = val_set & test_set
    if overlap:
        names = ", ".join(sorted(p.name for p in overlap))
        raise ValueError(f"Validation and test splits overlap: {names}")

    remaining = [p for p in files if p.resolve() not in (val_set | test_set)]

    if not test_files:
        if test_count > len(remaining):
            raise ValueError(f"test_count={test_count} exceeds remaining files={len(remaining)}")
        indices = sorted(rng.choice(len(remaining), size=test_count, replace=False).tolist())
        test = [remaining[i] for i in indices]
        test_set = {p.resolve() for p in test}
        remaining = [p for p in remaining if p.resolve() not in test_set]

    if not val_files:
        if val_count > len(remaining):
            raise ValueError(f"val_count={val_count} exceeds remaining files={len(remaining)}")
        indices = sorted(rng.choice(len(remaining), size=val_count, replace=False).tolist())
        val = [remaining[i] for i in indices]
        val_set = {p.resolve() for p in val}
        remaining = [p for p in remaining if p.resolve() not in val_set]

    return sorted(remaining), sorted(val), sorted(test)


def save_split(path: Path, train: Sequence[Path], test: Sequence[Path]) -> None:
    payload = {
        "train": [p.name for p in train],
        "test": [p.name for p in test],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def split_payload(
    train: Sequence[Path],
    val: Sequence[Path],
    test: Sequence[Path],
    seed: int | None = None,
    split_csv: str | Path | None = None,
) -> dict[str, object]:
    return {
        "source": "csv" if split_csv is not None else "seeded_random",
        "split_csv": str(Path(split_csv).expanduser().resolve()) if split_csv is not None else None,
        "seed": seed,
        "counts": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },
        "train": [p.name for p in train],
        "val": [p.name for p in val],
        "test": [p.name for p in test],
    }


def save_split_three_way(
    path: Path,
    train: Sequence[Path],
    val: Sequence[Path],
    test: Sequence[Path],
    seed: int | None = None,
    split_csv: str | Path | None = None,
) -> dict[str, object]:
    payload = split_payload(train, val, test, seed, split_csv)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_split(path: Path, all_files: Sequence[Path]) -> tuple[list[Path], list[Path]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return split_files(all_files, test_files=payload["test"])


def make_training_index(
    files: Sequence[Path],
    start_time: int = 1,
    end_time: int = 78,
) -> list[tuple[Path, int]]:
    index = []
    for path in files:
        for t in range(start_time, end_time + 1):
            index.append((path, t))
    return index


def summarize_files(files: Iterable[Path], boundary_percentile: float = 2.0) -> list[GraphMetadata]:
    cache = CaseCache(boundary_percentile)
    rows = []
    for path in files:
        static = cache.static(path)
        rows.append(
            GraphMetadata(
                case_name=case_id(path),
                path=path,
                n_nodes=int(static["points"].shape[0]),
                n_cells=int(static["cells"].shape[0]),
                n_edges=int(static["edge_index"].shape[1]),
            )
        )
    return rows
