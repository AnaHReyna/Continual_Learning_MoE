#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/../00_setup/env.sh"

python prototypes/train_teacher_task_encoder.py \
  --dataset-dirs \
    ../../export_rollouts/datasets/lane_keeping \
    ../../export_rollouts/datasets/change_lane_eval \
    ../../export_rollouts/datasets/pedestrian_antigo \
  --outdir teacher_task_encoder_ckpts_stage2 \
  --window-len 20 \
  --stride 10 \
  --batch-size 64 \
  --epochs 100 \
  --lr 1e-4 \
  --save-every 5 \
  --action-source teacher_mean_action \
  --reward-key rew \
  --balance-tasks \
  --model-dim 128 \
  --num-heads 4 \
  --ff-dim 256 \
  --num-layers 2 \
  --task-emb-dim 16 \
  --dropout 0.1 \
  --temperature 0.1 \
  --use-cls-token
