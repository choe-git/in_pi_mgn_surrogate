from __future__ import annotations

import argparse

import numpy as np

from gnn_surrogate.data import CaseCache, case_id, list_h5_files


def summarize_mask(points: np.ndarray, velocity: np.ndarray, mask: np.ndarray) -> str:
    count = int(mask.sum())
    if count == 0:
        return "count=0"
    centroid = points[mask].mean(axis=0)
    speed = np.linalg.norm(velocity[mask], axis=1)
    return (
        f"count={count} "
        f"centroid=({centroid[0]:.4g}, {centroid[1]:.4g}, {centroid[2]:.4g}) "
        f"mean_speed_t1={speed.mean():.4g}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--max-cases", type=int, default=5)
    args = parser.parse_args()

    files = list_h5_files(args.data_dir)[: args.max_cases]
    cache = CaseCache()
    for path in files:
        static = cache.static(path)
        velocity_t1 = cache.velocity(path, 1)
        points = static["points"]
        print(f"{case_id(path)} {path.name}")
        print(f"  wall   {summarize_mask(points, velocity_t1, static['wall_mask'])}")
        print(f"  inlet  {summarize_mask(points, velocity_t1, static['inlet_mask'])}")
        print(f"  outlet {summarize_mask(points, velocity_t1, static['outlet_mask'])}")


if __name__ == "__main__":
    main()
