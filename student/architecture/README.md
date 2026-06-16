# Continual Learning MoE for Autonomous Driving

This folder contains the student architecture, training scripts, prototype analysis, CARLA evaluation scripts, and visualization utilities used for continual learning experiments with a Mixture-of-Experts model.

## Project structure

- common/: shared dataset, task encoder, and MoE layers.
- stage1/: Stage 1 student model and training.
- stage2/: Stage 2 student model and training.
- prototypes/: teacher task encoder and prototype generation.
- analysis/: prototype, embedding, and expert analyses.
- test_CARLA/evaluation/: CARLA evaluation scripts.
- test_CARLA/graphics/: plotting scripts and generated figures.
- scripts/: reproducible command wrappers.
- docs/: environment, results, and command documentation.
- results/: training checkpoints and logs.

## Reproducibility workflow

### 1. Setup

Run:

    source scripts/00_setup/env.sh

### 2. Stage 1 training

Run:

    bash scripts/01_stage1/train_stage1_lk_cl.sh

### 3. Teacher prototypes

Run:

    bash scripts/02_prototypes/train_teacher_encoder_stage1.sh
    bash scripts/02_prototypes/build_teacher_prototypes_stage1.sh
    bash scripts/02_prototypes/train_teacher_encoder_stage2.sh
    bash scripts/02_prototypes/build_teacher_prototypes_stage2.sh

### 4. Stage 2 Phase 1

Run:

    bash scripts/03_stage2_phase1/train_phase1_FT.sh
    bash scripts/03_stage2_phase1/train_phase1_GeoInt.sh
    bash scripts/03_stage2_phase1/train_phase1_Proto_AW.sh
    bash scripts/03_stage2_phase1/train_phase1_Proto_RP.sh
    bash scripts/03_stage2_phase1/train_phase1_Ours.sh

### 5. Stage 2 Phase 2

Run:

    bash scripts/04_stage2_phase2/train_phase2_Ours_ckpt20.sh
    bash scripts/04_stage2_phase2/train_phase2_Proto_AW_ckpt14.sh

### 6. CARLA evaluation

Smoke test:

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

Full evaluation:

    bash scripts/05_evaluation/eval_all_existing_results.sh

By default, reproduced outputs are saved under:

    test_CARLA/Resultados_reproduced/

### 7. Graphics and visualizations

See:

    docs/graphics_guide.md
    docs/graphics_inventory.txt

Wrapper scripts should be placed under:

    scripts/06_graphics/

### 8. Results

Official CARLA results are stored under:

    test_CARLA/Resultados/

Training checkpoints are stored under:

    results/

For compact maps and inventories, see:

    docs/results_map.md
    docs/evaluation_inventory.txt
    docs/checkpoints_inventory.txt
    docs/scripts_inventory.txt

## Environment

See:

    docs/environment_snapshot.txt

Important runtime paths are configured in:

    scripts/00_setup/env.sh

## Notes

- The current CARLA evaluator does not use the argument --level.
- CARLA evaluation requires the CARLA server to be running.
- Large checkpoints, logs, embeddings, and reproduced outputs are ignored by .gitignore.
