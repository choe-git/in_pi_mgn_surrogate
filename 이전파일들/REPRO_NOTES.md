# Reproduction Notes

## Paper Targets

From the PDF text extraction:

- Dataset: BenchAnXplore coarse/light dataset.
- Cases: 105 geometries.
- Timesteps: 80 frames per case.
- Mesh size: under 25k nodes and 120k elements in the light dataset.
- Train/test: 95 train simulations and 10 random test simulations.
- Timestep: 0.01 s.
- Model: MeshGraphNet encode-process-decode.
- MLPs: two hidden layers, 128 neurons, ReLU.
- Latent size: 128.
- Message passing: 15 rounds.
- Learning rate: 1e-4 for 16 epochs, exponential decay to 1e-7 over 4 epochs.
- Training: 20 epochs, about 144k training steps in the paper.
- Reported best model: In-PI-MGN.

## Reported Metrics

Table 1, whole geometry, errors in mm/s:

| Model | Input | Loss | 1-RMSE | 50-RMSE | 50-RMSE SD |
| --- | --- | --- | ---: | ---: | ---: |
| MGN | u | data | 1.11 | 50.51 | 1.43 |
| PI-MGN | u | PI | 1.06 | 55.62 | 1.27 |
| In-MGN | u, du/dt, inflow | data | 1.09 | 9.22 | 1.47 |
| In-PI-MGN | u, du/dt, inflow | PI | 0.85 | 7.58 | 1.02 |

## Implementation Gaps To Tune

The paper does not publish source code and the exact test case filenames are
not listed in text. This implementation therefore uses:

- A deterministic random 95/10 split by default.
- A heuristic inlet/outlet detector because the released HDF5 files expose
  `wall_mask` but not explicit inlet/outlet masks.
- Physics-inspired graph finite-difference losses, because pressure and the
  original solver-side differential operators are not directly exposed in the
  released HDF5 fields.

These are the main levers to tune if the first full run misses the paper table:

1. Recover exact 10 test case IDs from Fig. 1 or authors.
2. Improve inlet/outlet masks from geometry planes or XDMF/mesh metadata.
3. Tune physics loss scaling after observing first-epoch data/physics loss
   magnitudes.
4. Match any hidden normalization used by the authors.

## Server State

Prepared on `rintern07`:

- Project: `/home/rintern07/gnn_surrogate_2026`
- Dataset: `/home/rintern07/04_npj/coarse_dataset`
- Uploaded zip: `/home/rintern07/04_npj/npj_2026_dataset.zip`
- Smoke MGN run: `/home/rintern07/gnn_surrogate_2026/runs/smoke_cpu4`
- Smoke In-PI-MGN run: `/home/rintern07/gnn_surrogate_2026/runs/smoke_in_pi_cpu`

Smoke tests were run on login-node CPU with tiny model settings only to validate
data loading, forward/backward, checkpointing, and metric code.

## First Full Run

Run directory on `rintern07`:

- `/home/rintern07/gnn_surrogate_2026/runs/in_pi_mgn_tmpenv_screen`

Settings:

- Variant: `in-pi-mgn`
- Epochs: 20
- Message passing steps: 15
- Latent/hidden size: 128
- Device: CUDA, A100 80GB interactive `coss_agpu` session

Completed at `2026-06-24 16:24:40 KST` with exit status 0.

Final metrics from `eval_50rmse.log`:

| Metric | Mean | SD |
| --- | ---: | ---: |
| 1-RMSE | 1.5079 | 0.3096 |
| 50-RMSE | 121.1590 | 14.2463 |

The one-step error is in the right order of magnitude, but the 50-step rollout
is much worse than the paper target. The next tuning target should be rollout
stability: exact split recovery, inlet/outlet masks, loss scaling, and rollout
training/regularization.
