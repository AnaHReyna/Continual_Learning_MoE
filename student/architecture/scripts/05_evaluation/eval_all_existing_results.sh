#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/../00_setup/env.sh"

HOST="${HOST:-localhost}"
PORT="${PORT:-2000}"
TM_PORT="${TM_PORT:-8000}"
EPISODES="${EPISODES:-50}"

# OUT_ROOT=test_CARLA/Resultados bash scripts/05_evaluation/eval_all_existing_results.sh
OUT_ROOT="${OUT_ROOT:-test_CARLA/Resultados_reproduced}"

EVAL_PY="test_CARLA/evaluation/evaluate_student_single_task_carla.py"

eval_stage1() {
  local task="$1"
  local outdir="$2"
  shift 2

  python "$EVAL_PY" \
    --model-type stage1_moe \
    --ckpt-dir results/stage1/stage1_moe_lane_change_ckpt20 \
    --ckpt-ids "$@" \
    --task "$task" \
    --eval-episodes "$EPISODES" \
    --eval-outdir "$outdir" \
    --task-dim 16 \
    --num-experts 2 \
    --host "$HOST" \
    --port "$PORT" \
    --traffic-manager-port "$TM_PORT"
}

eval_stage2_plain() {
  local ckpt_dir="$1"
  local task="$2"
  local outdir="$3"
  shift 3

  python "$EVAL_PY" \
    --model-type stage2_moe \
    --ckpt-dir "$ckpt_dir" \
    --ckpt-ids "$@" \
    --task "$task" \
    --eval-episodes "$EPISODES" \
    --eval-outdir "$outdir" \
    --task-dim 16 \
    --num-old-experts 2 \
    --num-new-experts 1 \
    --host "$HOST" \
    --port "$PORT" \
    --traffic-manager-port "$TM_PORT"
}

eval_stage2_geoint_mlp() {
  local ckpt_dir="$1"
  local task="$2"
  local outdir="$3"
  shift 3

  python "$EVAL_PY" \
    --model-type stage2_moe \
    --ckpt-dir "$ckpt_dir" \
    --ckpt-ids "$@" \
    --task "$task" \
    --eval-episodes "$EPISODES" \
    --eval-outdir "$outdir" \
    --task-dim 16 \
    --geo-dim 8 \
    --int-dim 8 \
    --geo-type mlp \
    --interaction-type mlp \
    --num-old-experts 2 \
    --num-new-experts 1 \
    --use-geo \
    --use-int \
    --host "$HOST" \
    --port "$PORT" \
    --traffic-manager-port "$TM_PORT"
}

eval_stage2_ours_cross() {
  local ckpt_dir="$1"
  local task="$2"
  local outdir="$3"
  shift 3

  python "$EVAL_PY" \
    --model-type stage2_moe \
    --ckpt-dir "$ckpt_dir" \
    --ckpt-ids "$@" \
    --task "$task" \
    --eval-episodes "$EPISODES" \
    --eval-outdir "$outdir" \
    --task-dim 16 \
    --geo-dim 8 \
    --int-dim 8 \
    --geo-type cross_attn \
    --interaction-type cross_attn \
    --num-old-experts 2 \
    --num-new-experts 1 \
    --use-geo \
    --use-int \
    --host "$HOST" \
    --port "$PORT" \
    --traffic-manager-port "$TM_PORT"
}

echo "============================================================"
echo "Stage 1"
echo "============================================================"

eval_stage1 lane_keeping "$OUT_ROOT/stage1/lane_keeping" \
  ckpt-10 ckpt-15 ckpt-20

eval_stage1 change_lane "$OUT_ROOT/stage1/change_lane" \
  ckpt-10 ckpt-15 ckpt-20


echo "============================================================"
echo "Baseline: Zero-shot Stage 1 ckpt-20 on pedestrian"
echo "============================================================"

eval_stage1 pedestrian "$OUT_ROOT/baselines/zero_shot_stage1_ckpt20/pedestrian" \
  ckpt-20


echo "============================================================"
echo "Baseline: FT no replay pedestrian"
echo "============================================================"

for task in lane_keeping change_lane pedestrian; do
  eval_stage2_plain \
    results/baselines/FT_no_replay_pedestrian \
    "$task" \
    "$OUT_ROOT/baselines/ft_no_replay_pedestrian/$task" \
    ckpt-26 ckpt-30 ckpt-35
done


echo "============================================================"
echo "Stage 2 Phase 1: FT"
echo "============================================================"

for task in lane_keeping change_lane pedestrian; do
  eval_stage2_plain \
    results/stage2/phase1/FT \
    "$task" \
    "$OUT_ROOT/stage2/phase1/FT/$task" \
    ckpt-10 ckpt-12 ckpt-14 ckpt-16 ckpt-18 ckpt-20
done


echo "============================================================"
echo "Stage 2 Phase 1: GeoInt"
echo "============================================================"

eval_stage2_geoint_mlp \
  results/stage2/phase1/GeoInt \
  change_lane \
  "$OUT_ROOT/stage2/phase1/GeoInt/change_lane" \
  ckpt-10 ckpt-12 ckpt-14 ckpt-16 ckpt-18 ckpt-20

eval_stage2_geoint_mlp \
  results/stage2/phase1/GeoInt \
  lane_keeping \
  "$OUT_ROOT/stage2/phase1/GeoInt/lane_keeping" \
  ckpt-14 ckpt-16 ckpt-18 ckpt-20

eval_stage2_geoint_mlp \
  results/stage2/phase1/GeoInt \
  pedestrian \
  "$OUT_ROOT/stage2/phase1/GeoInt/pedestrian" \
  ckpt-10 ckpt-12 ckpt-14 ckpt-16 ckpt-18 ckpt-20


echo "============================================================"
echo "Stage 2 Phase 1: Proto_AW"
echo "============================================================"

for task in lane_keeping change_lane pedestrian; do
  eval_stage2_plain \
    results/stage2/phase1/Proto_AW \
    "$task" \
    "$OUT_ROOT/stage2/phase1/Proto_AW/$task" \
    ckpt-10 ckpt-12 ckpt-14 ckpt-16 ckpt-18 ckpt-20
done


echo "============================================================"
echo "Stage 2 Phase 1: Proto_RP"
echo "============================================================"

for task in lane_keeping change_lane pedestrian; do
  eval_stage2_plain \
    results/stage2/phase1/Proto_RP \
    "$task" \
    "$OUT_ROOT/stage2/phase1/Proto_RP/$task" \
    ckpt-10 ckpt-12 ckpt-14 ckpt-16 ckpt-18 ckpt-20
done


echo "============================================================"
echo "Stage 2 Phase 1: Ours"
echo "============================================================"

for task in lane_keeping change_lane pedestrian; do
  eval_stage2_ours_cross \
    results/stage2/phase1/Ours \
    "$task" \
    "$OUT_ROOT/stage2/phase1/Ours/$task" \
    ckpt-14 ckpt-16 ckpt-18 ckpt-20
done


echo "============================================================"
echo "Stage 2 Phase 2: Ours"
echo "============================================================"

for task in lane_keeping change_lane pedestrian; do
  eval_stage2_ours_cross \
    results/stage2/phase2/Ours_ckpt20 \
    "$task" \
    "$OUT_ROOT/stage2/phase2/Ours/$task" \
    ckpt-8 ckpt-10 ckpt-14 ckpt-18
done


echo "============================================================"
echo "Stage 2 Phase 2: Proto_AW"
echo "============================================================"

for task in lane_keeping change_lane pedestrian; do
  eval_stage2_plain \
    results/stage2/phase2/Proto-AW_ckpt14 \
    "$task" \
    "$OUT_ROOT/stage2/phase2/Proto_AW/$task" \
    ckpt-8 ckpt-10 ckpt-14 ckpt-18
done

echo "[OK] All evaluations finished."
echo "[OK] Outputs saved under: $OUT_ROOT"
