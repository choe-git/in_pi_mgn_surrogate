from __future__ import annotations

import argparse
from statistics import mean

from gnn_surrogate.data import list_h5_files, resolve_data_dir, summarize_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--max-cases", type=int, default=5)
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    files = list_h5_files(data_dir)
    rows = summarize_files(files[: args.max_cases])
    print(f"data_dir={data_dir}")
    print(f"n_h5_files={len(files)}")
    for row in rows:
        print(
            f"{row.case_name}: nodes={row.n_nodes} cells={row.n_cells} "
            f"directed_edges={row.n_edges} file={row.path.name}"
        )
    if rows:
        print(f"mean_nodes_first_{len(rows)}={mean(r.n_nodes for r in rows):.1f}")
        print(f"mean_edges_first_{len(rows)}={mean(r.n_edges for r in rows):.1f}")


if __name__ == "__main__":
    main()
