# Graphics Guide

This project keeps visualization and plotting scripts under two main locations:

- `test_CARLA/graphics/`
- `analysis/`

The folder `scripts/06_graphics/` is reserved for reproducible wrapper scripts that call the plotting routines used to generate figures for reports, slides, and papers.

## Current graphics inventory

See:

- `docs/graphics_inventory.txt`

## Main graphics folders

- `test_CARLA/graphics/train/`
- `test_CARLA/graphics/experts/`
- `analysis/embeddings/`
- `analysis/prototypes/`
- `analysis/stage1/`

## Available plotting and analysis scripts

- `test_CARLA/graphics/experts/scripts/expert_usage_combined_horizontal.py`
- `test_CARLA/graphics/experts/scripts/stacked_plot_combined.py`
- `test_CARLA/graphics/experts/scripts/stacked_plot.py`
- `test_CARLA/graphics/train/scripts/train_graphic_combined.py`
- `test_CARLA/graphics/train/scripts/train_graphic.py`
- `analysis/embeddings/plot_tsne_embeddings.py`
- `analysis/prototypes/analyze_teacher_task_prototypes.py`
- `analysis/prototypes/analyze_teacher_task_prototypes_loo.py`
- `analysis/stage1/analyze_stage1_experts.py`

## Expected outputs

Generated figures are usually saved under:

- `test_CARLA/graphics/train/figures/`
- `test_CARLA/graphics/experts/figures/`

or inside the corresponding analysis output folder.

## Recommended workflow

1. Run or reproduce training.
2. Run CARLA evaluation.
3. Confirm that result files exist.
4. Run the corresponding plotting script.
5. Save generated figures under `test_CARLA/graphics/.../figures/`.
6. Document the generated figure path in the report, slides, or paper.

## Notes

The plotting scripts depend on result files generated under:

- `results/`
- `test_CARLA/Resultados/`

The folder `scripts/06_graphics/` should contain wrapper scripts only. The original plotting code remains under `test_CARLA/graphics/` and `analysis/`.
