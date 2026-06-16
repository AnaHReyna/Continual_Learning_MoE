import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


def l2_normalize(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (n + eps)


def cosine_matrix(x):
    x = l2_normalize(x)
    return x @ x.T


def pairwise_cosine_to_prototypes(embeddings, prototypes):
    emb = l2_normalize(embeddings)
    proto = l2_normalize(prototypes)
    return emb @ proto.T   # [N, K]


def pairwise_l2_to_prototypes(embeddings, prototypes):
    # [N, 1, D] - [1, K, D] -> [N, K]
    diff = embeddings[:, None, :] - prototypes[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def ensure_str_list(arr):
    out = []
    for x in arr:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return out


def plot_heatmap(mat, labels, title, out_path):
    plt.figure(figsize=(5, 4))
    plt.imshow(mat, interpolation="nearest")
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=20)
    plt.yticks(range(len(labels)), labels)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_histogram(correct_vals, wrong_vals, title, xlabel, out_path):
    plt.figure(figsize=(7, 5))
    plt.hist(correct_vals, bins=40, alpha=0.7, label="correct prototype")
    plt.hist(wrong_vals, bins=40, alpha=0.7, label="wrong prototype")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_scatter_2d(points, labels, task_names, prototypes_2d=None, out_path="plot.png", title="2D projection"):
    plt.figure(figsize=(7, 6))
    labels = np.asarray(labels)

    for tid in sorted(np.unique(labels)):
        idx = labels == tid
        name = task_names[int(tid)] if int(tid) < len(task_names) else f"task_{tid}"
        plt.scatter(points[idx, 0], points[idx, 1], s=8, alpha=0.65, label=name)

    if prototypes_2d is not None:
        for i in range(prototypes_2d.shape[0]):
            name = task_names[i] if i < len(task_names) else f"proto_{i}"
            plt.scatter(
                prototypes_2d[i, 0],
                prototypes_2d[i, 1],
                s=180,
                marker="X",
                edgecolors="black",
                linewidths=1.0,
                label=f"{name} prototype",
            )

    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def compute_confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def try_pca(X, n_components=2):
    X = np.asarray(X, dtype=np.float32)
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:n_components].T


def try_tsne(X, seed=42):
    try:
        from sklearn.manifold import TSNE
    except Exception:
        return None

    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0]
    if n < 10:
        return None

    perplexity = min(30, max(5, n // 100))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    return tsne.fit_transform(X)


def try_umap(X, seed=42):
    try:
        import umap
    except Exception:
        return None

    reducer = umap.UMAP(
        n_components=2,
        random_state=seed,
        n_neighbors=20,
        min_dist=0.1,
        metric="cosine",
    )
    return reducer.fit_transform(X)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proto-npz", type=str, required=True)
    parser.add_argument("--window-npz", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="teacher_task_analysis")
    parser.add_argument("--max-points-vis", type=int, default=4000)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    proto_data = np.load(args.proto_npz, allow_pickle=True)
    win_data = np.load(args.window_npz, allow_pickle=True)

    prototypes = proto_data["prototypes"].astype(np.float32)   # [K, D]
    proto_task_ids = proto_data["task_ids"].astype(np.int32)
    proto_task_names = ensure_str_list(proto_data["task_names"])

    embeddings = win_data["embeddings"].astype(np.float32)     # [N, D]
    labels = win_data["task_ids"].astype(np.int32)             # [N]

    # organizar task names por id
    max_tid = int(max(proto_task_ids.max(), labels.max()))
    task_names_by_id = [""] * (max_tid + 1)
    for tid, name in zip(proto_task_ids, proto_task_names):
        task_names_by_id[int(tid)] = name

    # matriz de similaridade entre prototypes
    proto_cos = cosine_matrix(prototypes)

    # similaridades embedding -> prototypes
    cos_to_proto = pairwise_cosine_to_prototypes(embeddings, prototypes)
    l2_to_proto = pairwise_l2_to_prototypes(embeddings, prototypes)

    pred_by_cos = np.argmax(cos_to_proto, axis=1)
    pred_by_l2 = np.argmin(l2_to_proto, axis=1)

    acc_cos = float((pred_by_cos == labels).mean())
    acc_l2 = float((pred_by_l2 == labels).mean())

    cm_cos = compute_confusion_matrix(labels, pred_by_cos, num_classes=prototypes.shape[0])
    cm_l2 = compute_confusion_matrix(labels, pred_by_l2, num_classes=prototypes.shape[0])

    # distância/similaridade correta vs incorreta
    correct_cos = cos_to_proto[np.arange(len(labels)), labels]
    wrong_cos = []
    correct_l2 = l2_to_proto[np.arange(len(labels)), labels]
    wrong_l2 = []

    for i in range(len(labels)):
        tid = labels[i]
        wrong_idx = [j for j in range(prototypes.shape[0]) if j != tid]
        wrong_cos.append(np.max(cos_to_proto[i, wrong_idx]))
        wrong_l2.append(np.min(l2_to_proto[i, wrong_idx]))

    wrong_cos = np.asarray(wrong_cos, dtype=np.float32)
    wrong_l2 = np.asarray(wrong_l2, dtype=np.float32)

    summary = {
        "num_embeddings": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "num_tasks": int(prototypes.shape[0]),
        "task_names": task_names_by_id,
        "prototype_shape": list(prototypes.shape),
        "prototype_cosine_similarity": proto_cos.tolist(),
        "prototype_offdiag_cosine_mean": float(
            proto_cos[~np.eye(proto_cos.shape[0], dtype=bool)].mean()
        ) if proto_cos.shape[0] > 1 else 1.0,
        "nearest_prototype_accuracy_cosine": acc_cos,
        "nearest_prototype_accuracy_l2": acc_l2,
        "mean_correct_cosine": float(correct_cos.mean()),
        "mean_wrong_cosine": float(wrong_cos.mean()),
        "mean_correct_l2": float(correct_l2.mean()),
        "mean_wrong_l2": float(wrong_l2.mean()),
        "confusion_matrix_cosine": cm_cos.tolist(),
        "confusion_matrix_l2": cm_l2.tolist(),
    }

    with open(os.path.join(args.outdir, "analysis_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # heatmaps
    plot_heatmap(
        proto_cos,
        labels=task_names_by_id[:prototypes.shape[0]],
        title="Prototype cosine similarity",
        out_path=os.path.join(args.outdir, "prototype_cosine_heatmap.png"),
    )

    plot_heatmap(
        cm_cos,
        labels=task_names_by_id[:prototypes.shape[0]],
        title="Confusion matrix (nearest prototype by cosine)",
        out_path=os.path.join(args.outdir, "confusion_cosine.png"),
    )

    plot_heatmap(
        cm_l2,
        labels=task_names_by_id[:prototypes.shape[0]],
        title="Confusion matrix (nearest prototype by L2)",
        out_path=os.path.join(args.outdir, "confusion_l2.png"),
    )

    # histograms
    plot_histogram(
        correct_cos,
        wrong_cos,
        title="Cosine similarity to correct vs wrong prototype",
        xlabel="Cosine similarity",
        out_path=os.path.join(args.outdir, "hist_cosine_correct_vs_wrong.png"),
    )

    plot_histogram(
        correct_l2,
        wrong_l2,
        title="L2 distance to correct vs wrong prototype",
        xlabel="L2 distance",
        out_path=os.path.join(args.outdir, "hist_l2_correct_vs_wrong.png"),
    )

    # reduzir pontos para visualização
    n = embeddings.shape[0]
    if n > args.max_points_vis:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=args.max_points_vis, replace=False)
        emb_vis = embeddings[idx]
        labels_vis = labels[idx]
    else:
        emb_vis = embeddings
        labels_vis = labels

    # PCA
    emb_pca = try_pca(emb_vis, n_components=2)
    proto_pca = try_pca(np.vstack([emb_vis, prototypes]), n_components=2)[-prototypes.shape[0]:]
    plot_scatter_2d(
        emb_pca,
        labels_vis,
        task_names_by_id,
        prototypes_2d=proto_pca,
        out_path=os.path.join(args.outdir, "pca_embeddings.png"),
        title="PCA of window embeddings",
    )

    # t-SNE
    emb_tsne = try_tsne(emb_vis)
    if emb_tsne is not None:
        proto_tsne = try_tsne(np.vstack([emb_vis, prototypes]))
        if proto_tsne is not None:
            proto_tsne = proto_tsne[-prototypes.shape[0]:]
        plot_scatter_2d(
            emb_tsne,
            labels_vis,
            task_names_by_id,
            prototypes_2d=proto_tsne,
            out_path=os.path.join(args.outdir, "tsne_embeddings.png"),
            title="t-SNE of window embeddings",
        )

    # UMAP
    emb_umap = try_umap(emb_vis)
    if emb_umap is not None:
        proto_umap = try_umap(np.vstack([emb_vis, prototypes]))
        if proto_umap is not None:
            proto_umap = proto_umap[-prototypes.shape[0]:]
        plot_scatter_2d(
            emb_umap,
            labels_vis,
            task_names_by_id,
            prototypes_2d=proto_umap,
            out_path=os.path.join(args.outdir, "umap_embeddings.png"),
            title="UMAP of window embeddings",
        )

    print("\nSaved analysis to:", args.outdir)
    print("Nearest prototype accuracy (cosine):", acc_cos)
    print("Nearest prototype accuracy (l2):", acc_l2)
    print("Mean correct cosine:", float(correct_cos.mean()))
    print("Mean wrong cosine:", float(wrong_cos.mean()))
    print("Mean correct l2:", float(correct_l2.mean()))
    print("Mean wrong l2:", float(wrong_l2.mean()))
    print("Prototype cosine similarity matrix:\n", proto_cos)


if __name__ == "__main__":
    main()


# python analyze_teacher_task_prototypes.py \
#   --proto-npz teacher_task_prototypes/teacher_task_prototypes.npz \
#   --window-npz teacher_task_prototypes/teacher_window_embeddings.npz \
#   --outdir teacher_task_analysis