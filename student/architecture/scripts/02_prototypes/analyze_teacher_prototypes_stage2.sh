#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/../00_setup/env.sh"

python analysis/prototypes/analyze_teacher_task_prototypes.py \
  --proto-npz teacher_task_prototypes/teacher_task_prototypes_stage2.npz \
  --window-npz teacher_task_prototypes/teacher_window_embeddings_stage2.npz \
  --outdir teacher_task_analysis_stage2 \
  --max-points-vis 4000

python analysis/prototypes/analyze_teacher_task_prototypes_loo.py \
  --proto-npz teacher_task_prototypes/teacher_task_prototypes_stage2.npz \
  --window-npz teacher_task_prototypes/teacher_window_embeddings_stage2.npz \
  --outdir teacher_task_analysis_stage2_loo
