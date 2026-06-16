import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


def l2_normalize(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (n + eps)


def ensure_str_list(arr):
    out = []
    for x in arr:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return out


def compute_confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


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


def plot_margin_histogram(margins, title, xlabel, out_path):
    plt.figure(figsize=(7, 5))
    plt.hist(margins, bins=40, alpha=0.8)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proto-npz", type=str, required=True)
    parser.add_argument("--window-npz", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="teacher_task_analysis_loo")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    proto_data = np.load(args.proto_npz, allow_pickle=True)
    win_data = np.load(args.window_npz, allow_pickle=True)

    task_ids_proto = proto_data["task_ids"].astype(np.int32)
    task_names_proto = ensure_str_list(proto_data["task_names"])

    embeddings = win_data["embeddings"].astype(np.float32)   # [N, D]
    labels = win_data["task_ids"].astype(np.int32)           # [N]

    embeddings = l2_normalize(embeddings)

    num_classes = len(task_ids_proto)
    dim = embeddings.shape[1]

    task_names_by_id = [""] * (labels.max() + 1)
    for tid, name in zip(task_ids_proto, task_names_proto):
        task_names_by_id[int(tid)] = name

    # soma e contagem por classe
    class_sums = np.zeros((num_classes, dim), dtype=np.float32)
    class_counts = np.zeros((num_classes,), dtype=np.int32)

    for z, y in zip(embeddings, labels):
        class_sums[int(y)] += z
        class_counts[int(y)] += 1

    if np.any(class_counts <= 1):
        raise ValueError(
            f"Each class must have at least 2 embeddings for leave-one-out. Counts={class_counts.tolist()}"
        )

    pred_cos = np.zeros((len(labels),), dtype=np.int32)
    pred_l2 = np.zeros((len(labels),), dtype=np.int32)

    correct_cos_vals = np.zeros((len(labels),), dtype=np.float32)
    wrong_cos_vals = np.zeros((len(labels),), dtype=np.float32)

    correct_l2_vals = np.zeros((len(labels),), dtype=np.float32)
    wrong_l2_vals = np.zeros((len(labels),), dtype=np.float32)

    for i in range(len(labels)):
        y = int(labels[i])
        z = embeddings[i]

        proto_list = []
        for c in range(num_classes):
            if c == y:
                proto_c = (class_sums[c] - z) / float(class_counts[c] - 1)
            else:
                proto_c = class_sums[c] / float(class_counts[c])

            proto_c = l2_normalize(proto_c)
            proto_list.append(proto_c)

        protos_i = np.stack(proto_list, axis=0)   # [K, D]

        cos_scores = protos_i @ z
        l2_scores = np.linalg.norm(protos_i - z[None, :], axis=1)

        pred_cos[i] = int(np.argmax(cos_scores))
        pred_l2[i] = int(np.argmin(l2_scores))

        correct_cos_vals[i] = float(cos_scores[y])

        wrong_idx = [c for c in range(num_classes) if c != y]
        wrong_cos_vals[i] = float(np.max(cos_scores[wrong_idx]))
        correct_l2_vals[i] = float(l2_scores[y])
        wrong_l2_vals[i] = float(np.min(l2_scores[wrong_idx]))

    acc_cos = float((pred_cos == labels).mean())
    acc_l2 = float((pred_l2 == labels).mean())

    cm_cos = compute_confusion_matrix(labels, pred_cos, num_classes)
    cm_l2 = compute_confusion_matrix(labels, pred_l2, num_classes)

    cosine_margins = correct_cos_vals - wrong_cos_vals
    l2_margins = wrong_l2_vals - correct_l2_vals

    summary = {
        "num_embeddings": int(len(labels)),
        "num_classes": int(num_classes),
        "task_names": task_names_by_id[:num_classes],
        "class_counts": class_counts.tolist(),
        "leave_one_out_accuracy_cosine": acc_cos,
        "leave_one_out_accuracy_l2": acc_l2,
        "mean_correct_cosine": float(correct_cos_vals.mean()),
        "mean_wrong_cosine": float(wrong_cos_vals.mean()),
        "mean_cosine_margin": float(cosine_margins.mean()),
        "mean_correct_l2": float(correct_l2_vals.mean()),
        "mean_wrong_l2": float(wrong_l2_vals.mean()),
        "mean_l2_margin": float(l2_margins.mean()),
        "confusion_matrix_cosine": cm_cos.tolist(),
        "confusion_matrix_l2": cm_l2.tolist(),
    }

    with open(os.path.join(args.outdir, "analysis_summary_loo.json"), "w") as f:
        json.dump(summary, f, indent=2)

    plot_heatmap(
        cm_cos,
        labels=task_names_by_id[:num_classes],
        title="Confusion matrix (leave-one-out cosine)",
        out_path=os.path.join(args.outdir, "confusion_loo_cosine.png"),
    )

    plot_heatmap(
        cm_l2,
        labels=task_names_by_id[:num_classes],
        title="Confusion matrix (leave-one-out L2)",
        out_path=os.path.join(args.outdir, "confusion_loo_l2.png"),
    )

    plot_histogram(
        correct_cos_vals,
        wrong_cos_vals,
        title="LOO cosine: correct vs wrong prototype",
        xlabel="Cosine similarity",
        out_path=os.path.join(args.outdir, "hist_loo_cosine_correct_vs_wrong.png"),
    )

    plot_histogram(
        correct_l2_vals,
        wrong_l2_vals,
        title="LOO L2: correct vs wrong prototype",
        xlabel="L2 distance",
        out_path=os.path.join(args.outdir, "hist_loo_l2_correct_vs_wrong.png"),
    )

    plot_margin_histogram(
        cosine_margins,
        title="LOO cosine margin",
        xlabel="correct_cosine - wrong_cosine",
        out_path=os.path.join(args.outdir, "hist_loo_cosine_margin.png"),
    )

    plot_margin_histogram(
        l2_margins,
        title="LOO L2 margin",
        xlabel="wrong_l2 - correct_l2",
        out_path=os.path.join(args.outdir, "hist_loo_l2_margin.png"),
    )

    print("\nSaved leave-one-out analysis to:", args.outdir)
    print("Leave-one-out accuracy (cosine):", acc_cos)
    print("Leave-one-out accuracy (l2):", acc_l2)
    print("Mean correct cosine:", float(correct_cos_vals.mean()))
    print("Mean wrong cosine:", float(wrong_cos_vals.mean()))
    print("Mean cosine margin:", float(cosine_margins.mean()))
    print("Mean correct l2:", float(correct_l2_vals.mean()))
    print("Mean wrong l2:", float(wrong_l2_vals.mean()))
    print("Mean l2 margin:", float(l2_margins.mean()))


if __name__ == "__main__":
    main()

# python analyze_teacher_task_prototypes_loo.py \
#   --proto-npz teacher_task_prototypes/teacher_task_prototypes.npz \
#   --window-npz teacher_task_prototypes/teacher_window_embeddings.npz \
#   --outdir teacher_task_analysis_loo