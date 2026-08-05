"""Build and sample a dataset-level inlet velocity profile on normalized coordinates.

This module is designed for the HDF5 layout used by this repository:

* ``data0``: node positions, shape ``[N, 3]``
* ``data1``: tetrahedra, shape ``[M, 4]``
* ``data2, data4, ..., data160``: velocity at 80 time steps

The profile coordinate system is case-independent: every inlet cap (plus the
first attached wall-node ring) is projected onto its local inlet plane and
scaled independently to ``[-1, 1]`` on both axes. Velocity magnitude is kept in
its original HDF5 units without normalization. Training case profiles selected
by ``split.csv`` are linearly interpolated to a common regular grid and averaged.

From the repository root, a typical build command is::

    PYTHONPATH=$PWD python -m gnn_surrogate.inlet_profile build \
        --data-dir 04_npj_GNN \
        --split-csv split.csv \
        --output gnn_surrogate/inlet_profile.npz

The file can also be run directly from ``gnn_surrogate``::

    python inlet_profile.py build --data-dir ../04_npj_GNN
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np

try:
    from .data import (
        CaseCache,
        N_TIMESTEPS,
        split_files_three_way,
        tetrahedra_boundary_faces,
        velocity_key,
    )
except ImportError:  # Allow ``python inlet_profile.py ...`` from this directory.
    from data import (
        CaseCache,
        N_TIMESTEPS,
        split_files_three_way,
        tetrahedra_boundary_faces,
        velocity_key,
    )


EPS = np.finfo(np.float64).eps


@dataclass(frozen=True)
class CaseInletProfile:
    """Normalized profile extracted from one HDF5 case."""

    path: Path
    node_ids: np.ndarray
    inlet_node_ids: np.ndarray
    wall_node_ids: np.ndarray
    normalized_xy: np.ndarray
    velocity_norm: np.ndarray
    center: np.ndarray
    basis_x: np.ndarray
    basis_y: np.ndarray
    outward_normal: np.ndarray
    xy_min: np.ndarray
    xy_max: np.ndarray


@dataclass(frozen=True)
class BuildReport:
    discovered_files: int
    used_files: tuple[str, ...]
    failed_files: tuple[tuple[str, str], ...]
    output: Path
    grid_size: int
    timesteps: int


def default_data_dir() -> Path:
    """Return ``<repository-root>/04_npj_GNN`` for the installed file layout."""
    return Path(__file__).resolve().parents[1] / "04_npj_GNN"


def default_output_path() -> Path:
    return Path(__file__).resolve().with_name("inlet_profile.npz")


def default_split_csv() -> Path:
    return Path(__file__).resolve().parents[1] / "split.csv"


def find_h5_files(data_dir: str | Path) -> list[Path]:
    """Recursively find all HDF5 cases below ``data_dir``."""
    root = Path(data_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {root}")
    files = sorted(path for path in root.rglob("*.h5") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No .h5 files found recursively below: {root}")
    return files


def validate_h5(path: str | Path, timesteps: int = N_TIMESTEPS) -> tuple[int, int]:
    """Validate required datasets and return ``(n_nodes, n_cells)``."""
    path = Path(path)
    with h5py.File(path, "r") as h5:
        required = ["data0", "data1", *[velocity_key(t) for t in range(timesteps)]]
        missing = [key for key in required if key not in h5]
        if missing:
            preview = ", ".join(missing[:8])
            raise ValueError(f"missing HDF5 datasets: {preview}")

        points = h5["data0"]
        cells = h5["data1"]
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"data0 must have shape [N, 3], got {points.shape}")
        if cells.ndim != 2 or cells.shape[1] != 4:
            raise ValueError(f"data1 must have shape [M, 4], got {cells.shape}")
        n_nodes = int(points.shape[0])
        if n_nodes < 4 or int(cells.shape[0]) < 1:
            raise ValueError("mesh is empty")

        sample_indices = np.linspace(0, timesteps - 1, min(5, timesteps), dtype=int)
        for t in np.unique(sample_indices):
            dataset = h5[velocity_key(int(t))]
            if dataset.shape != (n_nodes, 3):
                raise ValueError(
                    f"{velocity_key(int(t))} must have shape {(n_nodes, 3)}, "
                    f"got {dataset.shape}"
                )
    return n_nodes, int(cells.shape[0])


def _attached_wall_ring(
    cells: np.ndarray,
    inlet_mask: np.ndarray,
    wall_mask: np.ndarray,
) -> np.ndarray:
    """Find wall nodes sharing a boundary triangle with an inlet node."""
    boundary_faces = tetrahedra_boundary_faces(cells)
    touches_inlet = np.any(inlet_mask[boundary_faces], axis=1)
    if not np.any(touches_inlet):
        return np.empty(0, dtype=np.int64)
    candidates = np.unique(boundary_faces[touches_inlet].reshape(-1))
    return candidates[wall_mask[candidates]].astype(np.int64, copy=False)


def _local_frame(
    points: np.ndarray,
    inlet_node_ids: np.ndarray,
    mean_inlet_velocity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cap = points[inlet_node_ids].astype(np.float64, copy=False)
    center = cap.mean(axis=0)
    _, _, vh = np.linalg.svd(cap - center, full_matrices=False)
    normal = vh[-1]
    normal /= max(float(np.linalg.norm(normal)), EPS)

    # Velocity points into the domain, so an outward cap normal points opposite.
    if float(np.dot(normal, mean_inlet_velocity)) > 0.0:
        normal = -normal

    # Deterministic in-plane orientation: project the best global axis.
    global_axes = np.eye(3, dtype=np.float64)
    projected = global_axes - np.outer(global_axes @ normal, normal)
    lengths = np.linalg.norm(projected, axis=1)
    basis_x = projected[int(np.argmax(lengths))]
    basis_x /= max(float(np.linalg.norm(basis_x)), EPS)
    basis_y = np.cross(normal, basis_x)
    basis_y /= max(float(np.linalg.norm(basis_y)), EPS)
    return center, basis_x, basis_y, normal


def normalize_positions(
    points: np.ndarray,
    node_ids: np.ndarray,
    center: np.ndarray,
    basis_x: np.ndarray,
    basis_y: np.ndarray,
    xy_min: np.ndarray | None = None,
    xy_max: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project selected 3-D nodes and scale each planar axis to ``[-1, 1]``."""
    relative = points[node_ids].astype(np.float64, copy=False) - center
    xy = np.column_stack((relative @ basis_x, relative @ basis_y))
    lower = xy.min(axis=0) if xy_min is None else np.asarray(xy_min, dtype=np.float64)
    upper = xy.max(axis=0) if xy_max is None else np.asarray(xy_max, dtype=np.float64)
    span = upper - lower
    if np.any(span <= 1e-12):
        raise ValueError(f"degenerate inlet coordinate range: min={lower}, max={upper}")
    normalized = 2.0 * (xy - lower) / span - 1.0
    return normalized.astype(np.float32), lower, upper


def extract_case_profile(
    path: str | Path,
    cache: CaseCache | None = None,
    include_attached_wall: bool = True,
) -> CaseInletProfile:
    """Extract normalized coordinates and all 80 raw velocity-norm profiles."""
    path = Path(path).expanduser().resolve()
    validate_h5(path)
    cache = CaseCache() if cache is None else cache
    static = cache.static(path)
    points = static["points"]
    cells = static["cells"]
    inlet_mask = static["inlet_mask"].astype(bool, copy=False)
    wall_mask = static["wall_mask"].astype(bool, copy=False)
    inlet_node_ids = np.flatnonzero(inlet_mask).astype(np.int64)
    if inlet_node_ids.size < 3:
        raise ValueError(f"inlet detection returned only {inlet_node_ids.size} nodes")

    wall_node_ids = (
        _attached_wall_ring(cells, inlet_mask, wall_mask)
        if include_attached_wall
        else np.empty(0, dtype=np.int64)
    )
    node_ids = np.unique(np.concatenate((inlet_node_ids, wall_node_ids)))

    velocities = cache.velocities(path).astype(np.float64, copy=False)
    mean_inlet_velocity = velocities[:, inlet_node_ids].mean(axis=(0, 1))
    center, basis_x, basis_y, normal = _local_frame(
        points, inlet_node_ids, mean_inlet_velocity
    )
    normalized_xy, xy_min, xy_max = normalize_positions(
        points, node_ids, center, basis_x, basis_y
    )

    speed = np.linalg.norm(velocities[:, node_ids], axis=2)
    if np.any(~np.isfinite(speed)):
        raise ValueError("inlet velocity norm contains NaN or infinite values")

    return CaseInletProfile(
        path=path,
        node_ids=node_ids,
        inlet_node_ids=inlet_node_ids,
        wall_node_ids=wall_node_ids,
        normalized_xy=normalized_xy,
        velocity_norm=speed.astype(np.float32),
        center=center.astype(np.float32),
        basis_x=basis_x.astype(np.float32),
        basis_y=basis_y.astype(np.float32),
        outward_normal=normal.astype(np.float32),
        xy_min=xy_min.astype(np.float32),
        xy_max=xy_max.astype(np.float32),
    )


def _linear_interpolate_case(
    case: CaseInletProfile,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate one irregular profile onto a regular grid."""
    try:
        import matplotlib.tri as mtri
    except ImportError as exc:
        raise ImportError(
            "Building an inlet profile requires matplotlib (matplotlib.tri)."
        ) from exc

    triangulation = mtri.Triangulation(
        case.normalized_xy[:, 0], case.normalized_xy[:, 1]
    )
    gx, gy = np.meshgrid(grid_x, grid_y, indexing="xy")
    result = np.full(
        (case.velocity_norm.shape[0], grid_y.size, grid_x.size),
        np.nan,
        dtype=np.float32,
    )
    for time_index, values in enumerate(case.velocity_norm):
        interpolator = mtri.LinearTriInterpolator(triangulation, values)
        interpolated = interpolator(gx, gy)
        result[time_index] = np.ma.filled(interpolated, np.nan).astype(np.float32)
    return result


def build_canonical_profile(
    data_dir: str | Path,
    output: str | Path,
    grid_size: int = 129,
    include_attached_wall: bool = True,
    strict: bool = False,
    split_csv: str | Path | None = None,
) -> BuildReport:
    """Build an averaged canonical profile from the selected HDF5 cases.

    When ``split_csv`` is provided, only its train rows are used.
    Corrupt/incompatible cases are reported and skipped unless ``strict=True``.
    The output is written atomically after at least one valid case succeeds.
    """
    if grid_size < 3:
        raise ValueError("grid_size must be at least 3")
    discovered_files = find_h5_files(data_dir)
    files = discovered_files
    resolved_split_csv = None
    if split_csv is not None:
        resolved_split_csv = Path(split_csv).expanduser().resolve()
        files, _, _ = split_files_three_way(
            discovered_files,
            split_csv=resolved_split_csv,
        )
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    axis = np.linspace(-1.0, 1.0, grid_size, dtype=np.float32)

    sums = np.zeros((N_TIMESTEPS, grid_size, grid_size), dtype=np.float64)
    sums_sq = np.zeros_like(sums)
    counts = np.zeros((N_TIMESTEPS, grid_size, grid_size), dtype=np.uint32)
    used: list[str] = []
    failed: list[tuple[str, str]] = []
    for path in files:
        try:
            # Keep memory bounded as the dataset grows: CaseCache retains all 80
            # velocity fields, so its lifetime must be limited to one H5 case.
            cache = CaseCache()
            case = extract_case_profile(
                path, cache=cache, include_attached_wall=include_attached_wall
            )
            interpolated = _linear_interpolate_case(case, axis, axis)
            valid = np.isfinite(interpolated)
            clean = np.where(valid, interpolated, 0.0)
            sums += clean
            sums_sq += clean * clean
            counts += valid.astype(np.uint32)
            used.append(str(path))
        except Exception as exc:  # Preserve the audit trail for every bad case.
            failed.append((str(path), f"{type(exc).__name__}: {exc}"))
            if strict:
                raise RuntimeError(f"Failed to process {path}: {exc}") from exc

    if not used:
        details = "; ".join(f"{path}: {reason}" for path, reason in failed[:3])
        raise RuntimeError(f"No valid HDF5 cases were processed. {details}")

    mean = np.full_like(sums, np.nan)
    second_moment = np.full_like(sums_sq, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(sums, counts, out=mean, where=counts > 0)
        np.divide(sums_sq, counts, out=second_moment, where=counts > 0)
        variance = second_moment - mean * mean
    mean[counts == 0] = np.nan
    variance[counts == 0] = np.nan
    std = np.sqrt(np.maximum(variance, 0.0))
    metadata = {
        "format_version": 1,
        "coordinate_normalization": "per-case local inlet x/y min-max to [-1, 1]",
        "velocity_normalization": "none; original HDF5 velocity-norm units",
        "interpolation": "piecewise-linear triangular interpolation",
        "aggregation": "unweighted arithmetic mean over valid case coverage",
        "split": "train" if resolved_split_csv is not None else "all",
        "split_csv": str(resolved_split_csv) if resolved_split_csv is not None else None,
        "include_attached_wall": bool(include_attached_wall),
        "used_files": used,
        "failed_files": [{"path": p, "reason": r} for p, r in failed],
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            grid_x=axis,
            grid_y=axis,
            velocity_norm=mean.astype(np.float32),
            velocity_norm_std=std.astype(np.float32),
            coverage_count=counts,
            metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
        )
    temporary.replace(output)

    return BuildReport(
        discovered_files=len(discovered_files),
        used_files=tuple(used),
        failed_files=tuple(failed),
        output=output,
        grid_size=grid_size,
        timesteps=N_TIMESTEPS,
    )


class CanonicalInletProfile:
    """Loaded canonical profile with bilinear sampling for model inference."""

    def __init__(
        self,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        velocity_norm: np.ndarray,
        coverage_count: np.ndarray,
        metadata: dict[str, object],
    ) -> None:
        self.grid_x = np.asarray(grid_x, dtype=np.float32)
        self.grid_y = np.asarray(grid_y, dtype=np.float32)
        self.velocity_norm = np.asarray(velocity_norm, dtype=np.float32)
        self.coverage_count = np.asarray(coverage_count)
        self.metadata = metadata
        expected = (N_TIMESTEPS, self.grid_y.size, self.grid_x.size)
        if self.velocity_norm.shape != expected:
            raise ValueError(
                f"velocity_norm shape {self.velocity_norm.shape} != {expected}"
            )
        if self.coverage_count.shape != expected:
            raise ValueError(
                f"coverage_count shape {self.coverage_count.shape} != {expected}"
            )

        condition_statistics = []
        for time_index in range(N_TIMESTEPS):
            field = self.velocity_norm[time_index]
            valid = np.isfinite(field) & (self.coverage_count[time_index] > 0)
            values = field[valid]
            if values.size == 0:
                raise ValueError(
                    f"Canonical inlet profile has no valid coverage at time {time_index}"
                )
            condition_statistics.append(
                [float(values.mean()), float(values.min()), float(values.max())]
            )
        self._condition_statistics = np.asarray(condition_statistics, dtype=np.float32)

    @classmethod
    def load(cls, path: str | Path) -> "CanonicalInletProfile":
        path = Path(path).expanduser().resolve()
        with np.load(path, allow_pickle=False) as artifact:
            metadata = json.loads(str(artifact["metadata_json"].item()))
            return cls(
                grid_x=artifact["grid_x"],
                grid_y=artifact["grid_y"],
                velocity_norm=artifact["velocity_norm"],
                coverage_count=artifact["coverage_count"],
                metadata=metadata,
            )

    @property
    def timesteps(self) -> int:
        return int(self.velocity_norm.shape[0])

    def condition_statistics(self, time_index: int) -> np.ndarray:
        """Return canonical ``[mean, min, max]`` inlet speed at one time step."""
        if time_index < 0 or time_index >= self.timesteps:
            raise IndexError(f"time_index must be in [0, {self.timesteps - 1}]")
        return self._condition_statistics[time_index].copy()

    def sample(
        self,
        normalized_xy: np.ndarray,
        time_indices: int | Sequence[int] | None = None,
        fill_value: float = 0.0,
    ) -> np.ndarray:
        """Bilinearly sample normalized coordinates.

        Returns shape ``[T, N]`` for multiple/all time steps and ``[N]`` for a
        scalar ``time_indices``. Grid cells without dataset coverage use
        ``fill_value`` (zero is appropriate for attached wall nodes).
        """
        xy = np.asarray(normalized_xy, dtype=np.float64)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError(f"normalized_xy must have shape [N, 2], got {xy.shape}")

        scalar_time = np.isscalar(time_indices) and time_indices is not None
        if time_indices is None:
            indices = np.arange(self.timesteps, dtype=np.int64)
        elif scalar_time:
            indices = np.asarray([int(time_indices)], dtype=np.int64)
        else:
            indices = np.asarray(list(time_indices), dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= self.timesteps):
            raise IndexError(f"time indices must be in [0, {self.timesteps - 1}]")

        x = np.clip(xy[:, 0], self.grid_x[0], self.grid_x[-1])
        y = np.clip(xy[:, 1], self.grid_y[0], self.grid_y[-1])
        fx = (x - self.grid_x[0]) / (self.grid_x[-1] - self.grid_x[0])
        fy = (y - self.grid_y[0]) / (self.grid_y[-1] - self.grid_y[0])
        fx *= self.grid_x.size - 1
        fy *= self.grid_y.size - 1
        x0 = np.minimum(np.floor(fx).astype(np.int64), self.grid_x.size - 2)
        y0 = np.minimum(np.floor(fy).astype(np.int64), self.grid_y.size - 2)
        x1, y1 = x0 + 1, y0 + 1
        wx, wy = fx - x0, fy - y0

        field = self.velocity_norm[indices]
        v00 = field[:, y0, x0]
        v10 = field[:, y0, x1]
        v01 = field[:, y1, x0]
        v11 = field[:, y1, x1]
        sampled = (
            v00 * (1.0 - wx) * (1.0 - wy)
            + v10 * wx * (1.0 - wy)
            + v01 * (1.0 - wx) * wy
            + v11 * wx * wy
        )
        sampled = np.where(np.isfinite(sampled), sampled, fill_value)
        sampled = np.maximum(sampled, 0.0)
        result = sampled.astype(np.float32)
        return result[0] if scalar_time else result

    def sample_case(
        self,
        h5_path: str | Path,
        time_indices: int | Sequence[int] | None = None,
        as_velocity_vectors: bool = False,
        include_attached_wall: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Normalize an H5 inlet mesh and sample this profile onto its nodes.

        Returns ``(node_ids, values)``. Values are speed magnitudes unless
        ``as_velocity_vectors=True``, in which case they point along the inward
        cap normal and have shape ``[..., N, 3]``.
        """
        case = extract_case_profile(
            h5_path, include_attached_wall=include_attached_wall
        )
        values = self.sample(
            case.normalized_xy,
            time_indices=time_indices,
        )
        if as_velocity_vectors:
            values = values[..., None] * (-case.outward_normal)
        return case.node_ids, values.astype(np.float32)


def _build_command(args: argparse.Namespace) -> int:
    report = build_canonical_profile(
        data_dir=args.data_dir,
        output=args.output,
        grid_size=args.grid_size,
        include_attached_wall=not args.cap_only,
        strict=args.strict,
        split_csv=args.split_csv,
    )
    print(f"Discovered H5 files: {report.discovered_files}")
    print(f"Successfully used:    {len(report.used_files)}")
    print(f"Failed/skipped:       {len(report.failed_files)}")
    for path, reason in report.failed_files:
        print(f"  SKIP {path}: {reason}")
    print(f"Saved profile:        {report.output}")
    return 0


def _inspect_command(args: argparse.Namespace) -> int:
    profile = CanonicalInletProfile.load(args.profile)
    print(f"Profile:      {Path(args.profile).expanduser().resolve()}")
    print(f"Grid:         {profile.grid_x.size} x {profile.grid_y.size}")
    print(f"Time steps:   {profile.timesteps}")
    print(f"Cases used:   {len(profile.metadata.get('used_files', []))}")
    print(f"Cases failed: {len(profile.metadata.get('failed_files', []))}")
    print(f"Velocity norm range: min={np.nanmin(profile.velocity_norm):.6g}, "
          f"max={np.nanmax(profile.velocity_norm):.6g}")
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="build a canonical inlet profile")
    build.add_argument("--data-dir", type=Path, default=default_data_dir())
    build.add_argument("--output", type=Path, default=default_output_path())
    build.add_argument(
        "--split-csv",
        type=Path,
        default=default_split_csv(),
        help="Use only CSV rows assigned to train (default: repository split.csv)",
    )
    build.add_argument("--grid-size", type=int, default=129)
    build.add_argument("--cap-only", action="store_true", help="exclude attached wall nodes")
    build.add_argument("--strict", action="store_true", help="stop at the first bad H5 file")
    build.set_defaults(func=_build_command)

    inspect = commands.add_parser("inspect", help="inspect a saved profile artifact")
    inspect.add_argument("--profile", type=Path, default=default_output_path())
    inspect.set_defaults(func=_inspect_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
