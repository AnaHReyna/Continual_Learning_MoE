# Generated Artifacts Policy

This repository tracks source code, reproducibility scripts, configuration summaries, and documentation.

The following folders are intentionally not versioned because they contain generated outputs, checkpoints, binary arrays, logs, or CARLA evaluation results:

- results/
- teacher_task_encoder_ckpts/
- teacher_task_encoder_ckpts_stage2/
- teacher_task_prototypes/
- teacher_task_prototypes_stage2/
- test_CARLA/Resultados/
- test_CARLA/Resultados_reproduced/
- test_CARLA/Resultados_smoke/

These artifacts can be regenerated using the scripts under:

- scripts/01_stage1/
- scripts/02_prototypes/
- scripts/03_stage2_phase1/
- scripts/04_stage2_phase2/
- scripts/05_evaluation/

The documentation under docs/ records the expected locations, configurations, checkpoints, and evaluation results used in the experiments.

If exact trained checkpoints or generated prototype files are required, they should be stored externally, for example in institutional storage, Google Drive, Zenodo, or a GitHub Release, rather than committed directly to the repository.
