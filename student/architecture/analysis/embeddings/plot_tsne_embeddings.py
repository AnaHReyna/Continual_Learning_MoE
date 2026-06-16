import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
# import sys
# sys.path.append('../../../')



def load_npz(npz_path):
    data = np.load(npz_path)
    return {"task_embedding": data["task_embedding"],
            "mode_embedding": data["mode_embedding"],
            "expert_id": data["expert_id"],
            "task_id": data["task_id"],
            "mode_label": data["mode_label"],
            }


def maybe_subsample(arrays, max_points=5000, seed=42):
    n = len(next(iter(arrays.values())))
    if n <= max_points:
        return arrays

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    return {k: v[idx] for k, v in arrays.items()}


def tsne_2d(x, perplexity=30, seed=42):
    tsne = TSNE(n_components=2,
                perplexity=perplexity,
                random_state=seed,
                init="pca",
                learning_rate="auto",
                )
    return tsne.fit_transform(x)


def plot_scatter(points, labels, title, out_path, label_names=None):
    plt.figure(figsize=(8, 6))
    unique = np.unique(labels)

    for u in unique:
        mask = labels == u
        name = str(u)
        if label_names is not None and int(u) in label_names:
            name = label_names[int(u)]
        plt.scatter(points[mask, 0], points[mask, 1], s=8, alpha=0.65, label=name)

    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(markerscale=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, required=True, help="Path to epoch_XXX_embeddings.npz")
    parser.add_argument("--outdir", type=str, default="tsne_plots")
    parser.add_argument("--max-points", type=int, default=5000)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    arrays = load_npz(args.npz)
    arrays = maybe_subsample(arrays, max_points=args.max_points, seed=args.seed)

    task_names = {0: "pedestrian",
                  1: "lane_keeping",
                 } 
    
    mode_names = { 0: "straight",
                   1: "curve",
                 }

    mode_xy = tsne_2d(arrays["mode_embedding"], perplexity=args.perplexity, seed=args.seed)
    plot_scatter(mode_xy,
                 arrays["mode_label"],
                 "t-SNE of mode_embedding colored by mode_label",
                 os.path.join(args.outdir, "tsne_mode_by_mode_label.png"),
                 label_names=mode_names,
                )

    task_xy = tsne_2d(arrays["task_embedding"], perplexity=args.perplexity, seed=args.seed)
    plot_scatter(task_xy,
                 arrays["task_id"],
                 "t-SNE of task_embedding colored by task_id",
                 os.path.join(args.outdir, "tsne_task_by_task_id.png"),
                 label_names=task_names,
                )

    plot_scatter(mode_xy,
                 arrays["expert_id"],
                 "t-SNE of mode_embedding colored by expert_id",
                 os.path.join(args.outdir, "tsne_mode_by_expert_id.png"),
                 label_names=None,
                )

    plot_scatter(task_xy,
                 arrays["expert_id"],
                 "t-SNE of task_embedding colored by expert_id",
                 os.path.join(args.outdir, "tsne_task_by_expert_id.png"),
                 label_names=None,
                )

    print("Saved plots to:", os.path.abspath(args.outdir))
    for name in sorted(os.listdir(args.outdir)):
        print(" -", name)


if __name__ == "__main__":
    main()


# python3 plot_tsne_embeddings.py \
#   --npz Arquitetura/checkpoints_zint/stage2_moe_ped_phase1_zint/embeddings/epoch_060_embeddings.npz \
#   --outdir Arquitetura/checkpoints_zint/stage2_moe_ped_phase1_zint/tsne_epoch060


# com menos pontos
# python3 plot_tsne_embeddings.py \
#   --npz Arquitetura/checkpoints_zint/stage2_moe_ped_phase1_zint/embeddings/epoch_060_embeddings.npz \
#   --outdir Arquitetura/checkpoints_zint/stage2_moe_ped_phase1_zint/tsne_epoch060 \
#   --max-points 3000 \
#   --perplexity 30