# in_pi_mgn_surrogate
replication of GNN surrogate model for hemodynamcis

link of the paper trying to replicate is below 

https://www.nature.com/articles/s41746-026-02404-z

arguments for train, val, test are as below

Build the canonical inlet profile once before training:

>PYTHONPATH=$PWD python -m gnn_surrogate.inlet_profile build \
  --data-dir 04_npj_GNN/coarse_dataset \
  --split-csv split.csv \
  --output gnn_surrogate/inlet_profile.npz

## train
>PYTHONPATH=$PWD python scripts/train.py \
  --data-dir 04_npj_GNN/coarse_dataset \
  --split-csv split.csv \
  --inlet-profile gnn_surrogate/inlet_profile.npz \
  --output-dir "output_dir" \
  --model-variant in-pi-mgn \
  --epochs 200 \
  --message-passing-steps 15 \
  --lr 1e-4 \
  --lr-decay-start-epoch 16 \
  --lr-min 1e-7 \
  --noise-std 0.003 \
  --velocity-augmentations 1\
  --temporal-noise-correlation 0.8 \
  --spatial-noise-smoothing-steps 2 \
  --magnitude-jitter-std 0.02 \
  --best-rmse-steps 50 \
  --device cuda

## val
>PYTHONPATH=$PWD python scripts/evaluate.py \
  --data-dir 04_npj_GNN/coarse_dataset \
  --checkpoint "output_dir/log_dir"/best.pt \
  --inlet-profile gnn_surrogate/inlet_profile.npz \
  --split val \
  --rollout-steps 50 \
  --device cuda

## test
>PYTHONPATH=$PWD python scripts/evaluate.py \
  --data-dir 04_npj_GNN/coarse_dataset \
  --checkpoint "output_dir/log_dir"/best.pt \
  --inlet-profile gnn_surrogate/inlet_profile.npz \
  --split test \
  --rollout-steps 50 \
  --device cuda

## tip
when file not found error, try this command

>unset PYTHONPATH \
 export PYTHONPATH="$(pwd)"
