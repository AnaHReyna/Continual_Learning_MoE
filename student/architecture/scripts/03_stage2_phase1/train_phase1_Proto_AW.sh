#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/../00_setup/env.sh"

python stage2/train_student_moe_stage2.py \
  --dataset-dirs ../../export_rollouts/datasets/pedestrian_antigo \
  --replay-dirs \
    ../../export_rollouts/datasets/lane_keeping_replay_10pct_success_curve \
    ../../export_rollouts/datasets/change_lane_replay_10pct_success \
  --init-ckpt results/stage1/stage1_moe_lane_change_ckpt20/ckpt-20 \
  --phase 1 \
  --student-logdir results/stage2/phase1 \
  --student-name Proto_AW \
  --student-epochs 100 \
  --student-batch-size 64 \
  --student-lr 1e-4 \
  --student-target-mode mean_action \
  --num-old-experts 2 \
  --num-new-experts 1 \
  --task-dim 16 \
  --init-new-expert-from-old-avg \
  --new-expert-noise-std 0.001 \
  --teacher-task-prototypes teacher_task_prototypes/teacher_task_prototypes_stage2.npz \
  --use-task-contrastive \
  --task-contrastive-weight 0.005 \
  --task-contrastive-temp 0.1 \
  --use-task-alignment \
  --task-alignment-weight 0.01 \
  --router-balance-weight 0.01 \
  --router-balance-initial-weight 0.05 \
  --router-balance-final-weight 0.01 \
  --router-balance-warmup-epochs 20 \
  --router-entropy-weight 0.0 \
  --router-entropy-initial-weight 0.005 \
  --router-entropy-final-weight 0.0 \
  --router-entropy-warmup-epochs 20 \
  --steer-weight 2.0 \
  --speed-weight 1.0
