# Evaluation Guide

The main CARLA evaluation script is:

    scripts/05_evaluation/eval_all_existing_results.sh

By default, reproduced results are saved under:

    test_CARLA/Resultados_reproduced/

This avoids overwriting official results.

To run a smoke test with one episode:

    python test_CARLA/evaluation/evaluate_student_single_task_carla.py \
      --model-type stage1_moe \
      --ckpt-dir results/stage1/stage1_moe_lane_change_ckpt20 \
      --ckpt-ids ckpt-10 \
      --task lane_keeping \
      --eval-episodes 1 \
      --eval-outdir test_CARLA/Resultados_smoke/stage1/lane_keeping \
      --task-dim 16 \
      --num-experts 2 \
      --host localhost \
      --port 2000 \
      --traffic-manager-port 8000

To run the full reproduced evaluation:

    bash scripts/05_evaluation/eval_all_existing_results.sh

All official evaluations listed in docs/evaluation_inventory.txt use 50 episodes.

Important: the current evaluator does not use --level.
