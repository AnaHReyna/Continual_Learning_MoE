#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/../00_setup/env.sh"

python stage1/train_student_moe.py \
  --dataset-dirs \
    ../../export_rollouts/datasets/lane_keeping \
    ../../export_rollouts/datasets/change_lane_eval \
  --teacher-task-prototypes teacher_task_prototypes/teacher_task_prototypes.npz \
  --student-logdir results/stage1 \
  --student-name stage1_moe_lane_change_ckpt20 \
  --student-epochs 200 \
  --student-batch-size 64 \
  --student-lr 1e-4 \
  --student-save-every 10 \
  --student-target-mode mean_action \
  --num-experts 2 \
  --task-dim 16 \
  --router-balance-weight 0.01 \
  --steer-weight 2.0 \
  --speed-weight 1.0 \
  --expert-diversity-weight 0.0005 \
  --expert-diversity-warmup-epochs 5 \
  --expert-diversity-ramp-epochs 10 \
  --use-task-contrastive \
  --task-contrastive-weight 0.01 \
  --task-contrastive-temp 0.1 \
  --use-task-alignment \
  --task-alignment-weight 0.01
