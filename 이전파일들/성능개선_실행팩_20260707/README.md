# GNN Surrogate 2026 Reproduction

This is a close, runnable reimplementation of the MeshGraphNet-style model
described in:

Physics constrained graph neural network for real time prediction of
intracranial aneurysm hemodynamics, npj Digital Medicine, 2026.

The original source code is not public. This repository focuses on reproducing
the reported setup as closely as possible from the paper and the released
BenchAnXplore coarse dataset:

- 105 aneurysm geometries, 80 timesteps each.
- HDF5 layout: `data0` node coordinates, `data1` tetrahedra,
  `data2,data4,...,data160` velocity fields, and odd datasets for `wall_mask`.
- MeshGraphNet encoder-process-decoder model.
- 2 hidden-layer MLPs with 128 hidden units.
- 128 latent features.
- 15 message passing blocks by default.
- One-step velocity update prediction.
- Optional In-MGN node features: acceleration and inflow context.
- Optional physics-inspired regularization.
- 1-step and 50-step rollout RMSE in mm/s.

The paper's exact test-case IDs are only shown visually in Fig. 1 and not listed
in text. By default this code creates a deterministic 95/10 split from sorted
HDF5 filenames. If you recover the exact paper test files, pass them with
`--test-files`.

## Setup

```bash
cd ~/gnn_surrogate_2026
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If CUDA PyTorch is already installed on the server, use that environment instead
and install only missing packages such as `h5py` and `tqdm`.

## Inspect Data

```bash
python scripts/inspect_dataset.py --data-dir /path/to/04_npj/coarse_dataset
```

The data directory can be either the `coarse_dataset` folder or its parent.

## Train

Small smoke test:

```bash
python scripts/train.py \
  --data-dir /path/to/04_npj/coarse_dataset \
  --output-dir runs/smoke \
  --max-train-samples 20 \
  --max-test-cases 1 \
  --message-passing-steps 2 \
  --epochs 1
```

Paper-like In-PI-MGN run:

```bash
python scripts/train.py \
  --data-dir /path/to/04_npj/coarse_dataset \
  --output-dir runs/in_pi_mgn \
  --model-variant in-pi-mgn \
  --epochs 20 \
  --message-passing-steps 15 \
  --lr 1e-4 \
  --lr-decay-start-epoch 16 \
  --lr-min 1e-7 \
  --noise-std 0.003 \
  --device cuda
```

## Evaluate

```bash
python scripts/evaluate.py \
  --data-dir /path/to/04_npj/coarse_dataset \
  --checkpoint runs/in_pi_mgn/best.pt \
  --rollout-steps 50 \
  --device cuda
```

The target paper values for the best model are:

- In-PI-MGN 1-RMSE: 0.85 mm/s
- In-PI-MGN 50-RMSE: 7.58 mm/s
- In-PI-MGN 50-RMSE SD: 1.02 mm/s

The first run will likely differ until the exact split, preprocessing, inferred
inlet/outlet masks, and loss weights are tuned.
