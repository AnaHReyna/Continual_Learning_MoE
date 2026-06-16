#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/../00_setup/env.sh"

python prototypes/build_teacher_task_prototypes.py \
  --dataset-dirs \
    ../../export_rollouts/datasets/lane_keeping \
    ../../export_rollouts/datasets/change_lane_eval \
  --ckpt-dir teacher_task_encoder_ckpts \
  --ckpt-id ckpt-20 \
  --outdir teacher_task_prototypes \
  --window-len 20 \
  --stride 10 \
  --batch-size 64 \
  --action-source teacher_mean_action \
  --reward-key rew \
  --model-dim 128 \
  --num-heads 4 \
  --ff-dim 256 \
  --num-layers 2 \
  --task-emb-dim 16 \
  --dropout 0.1 \
  --use-cls-token \
  --save-window-embeddings
