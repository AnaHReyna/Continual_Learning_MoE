import os
import sys
sys.path.append('../../')

import csv
import json
import argparse

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from train.init_configs import get_argument, set_configs
from common.student_dataset import DistillTransitionDataset
from stage1.student_model_moe import StudentMoE


def build_student_params(args, algo_params):
    p = dict(algo_params.get("params", {}))
    p.setdefault("units", 128)
    p.setdefault("state_input", False)
    p.setdefault("LSTM", False)
    p.setdefault("cnn_lstm", False)
    p.setdefault("bptt", False)
    p.setdefault("ego_surr", False)
    p.setdefault("neighbours", getattr(args, "neighbors", 5))
    p.setdefault("time_step", getattr(args, "N_steps", 10))
    p.setdefault("make_rotation", True)
    p.setdefault("use_map", True)
    p.setdefault("num_traj", 1)
    p.setdefault("cnn", False)
    p.setdefault("path_length", 10)
    p.setdefault("head_num", 2)
    p.setdefault("use_hier", True)
    p.setdefault("random_aug", False)
    p.setdefault("no_ego_fut", False)
    p.setdefault("no_neighbor_fut", False)
    p.setdefault("carla", True)

    p["use_vision"] = bool(args.student_use_vision)
    p["vision_dim"] = int(args.student_vision_dim)
    p["fusion_type"] = args.student_fusion_type

    if "neighbors" in p and "neighbours" not in p:
        p["neighbours"] = p["neighbors"]
    return p


def batch_value_or_none(batch, key: str):
    x = batch[key]
    if len(x.shape) == 2 and x.shape[-1] == 0:
        return None
    if len(x.shape) == 1 and x.shape[-1] == 0:
        return None
    return tf.convert_to_tensor(x, dtype=tf.float32)


def restore_student(student, ckpt_path):
    ckpt = tf.train.Checkpoint(student=student)
    if tf.io.gfile.isdir(ckpt_path):
        latest = tf.train.latest_checkpoint(ckpt_path)
        if latest is None:
            raise ValueError(f"No checkpoint found inside directory: {ckpt_path}")
        ckpt_path = latest
    ckpt.restore(ckpt_path).expect_partial()
    print(f"Restored student checkpoint: {ckpt_path}")
    return ckpt_path


def ensure_expert_tensor(expert_outs):
    """
    Returns expert_outs as [B, N, D]
    """
    if isinstance(expert_outs, (list, tuple)):
        expert_outs = tf.stack(expert_outs, axis=1)
    return tf.convert_to_tensor(expert_outs, dtype=tf.float32)


def cosine_similarity_matrix(expert_outs_np):
    """
    expert_outs_np: [M, N, D]
    Returns mean cosine similarity matrix [N, N]
    """
    x = expert_outs_np / (np.linalg.norm(expert_outs_np, axis=-1, keepdims=True) + 1e-8)
    sims = np.einsum("mnd,mkd->mnk", x, x)  # [M, N, N]
    return sims.mean(axis=0)


def save_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def maybe_run_tsne(X, seed=42, perplexity=30):
    """
    Returns 2D embedding or None if sklearn is unavailable.
    """
    try:
        from sklearn.manifold import TSNE
    except Exception:
        return None

    n = X.shape[0]
    if n < 5:
        return None

    perplexity = min(perplexity, max(2, n // 10))
    tsne = TSNE(n_components=2,
                perplexity=perplexity,
                init="pca",
                learning_rate="auto",
                random_state=seed,
              )
    return tsne.fit_transform(X)


def plot_scatter_2d(X2, labels, out_path, title, label_prefix):
    if X2 is None:
        return

    plt.figure(figsize=(7, 6))
    labels = np.asarray(labels)
    for k in sorted(np.unique(labels)):
        idx = labels == k
        plt.scatter(X2[idx, 0], X2[idx, 1], s=8, alpha=0.7, label=f"{label_prefix}{k}")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_heatmap(mat, out_path, title):
    plt.figure(figsize=(5, 4))
    plt.imshow(mat, interpolation="nearest")
    plt.colorbar()
    plt.title(title)
    plt.xticks(range(mat.shape[1]))
    plt.yticks(range(mat.shape[0]))
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    parser = get_argument()
    parser.add_argument("--dataset-dirs", nargs="+", required=True)
    parser.add_argument("--ckpt-dir", type=str, required=True)
    parser.add_argument("--ckpt-id", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="expert_analysis_stage1")

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=-1)

    parser.add_argument("--num-experts", type=int, default=2)
    parser.add_argument("--task-dim", type=int, default=16)

    parser.add_argument("--student-target-mode", choices=["mean_action", "raw_mean"], default="mean_action")
    parser.add_argument("--student-use-vision", action="store_true")
    parser.add_argument("--student-vision-dim", type=int, default=280)
    parser.add_argument("--student-fusion-type", choices=["cross", "self"], default="cross")

    args = parser.parse_args()
    args, algo_params, _ = set_configs(args, test=True)

    gpus = tf.config.experimental.list_physical_devices("GPU")
    if args.gpu is not None and args.gpu >= 0 and len(gpus) > args.gpu:
        tf.config.set_visible_devices([gpus[args.gpu]], "GPU")
        tf.config.experimental.set_memory_growth(gpus[args.gpu], True)
        print("Using GPU:", gpus[args.gpu])
    else:
        tf.config.set_visible_devices([], "GPU")
        print("Running on CPU.")

    os.makedirs(args.outdir, exist_ok=True)

    dataset = DistillTransitionDataset(
        dataset_dirs=args.dataset_dirs,
        target_mode=args.student_target_mode,
        use_vision=args.student_use_vision,
        require_map=True,
    )

    ds = dataset.make_tf_dataset(
        batch_size=args.batch_size,
        shuffle=False,
        shuffle_buffer=1,
        repeat=False,
    )

    sample0 = dataset.get_specimen()
    state_shape = sample0["obs"].shape
    action_dim = sample0["act"].shape[-1]

    params = build_student_params(args, algo_params)

    student = StudentMoE(
        params=params,
        state_shape=state_shape,
        action_dim=action_dim,
        max_action=1.0,
        num_experts=args.num_experts,
        task_dim=args.task_dim,
    )

    ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_id)
    restored_ckpt = restore_student(student, ckpt_path)

    all_gate_probs = []
    all_expert_ids = []
    all_task_emb = []
    all_backbone = []
    all_expert_outs = []

    rows = []
    batch_count = 0

    for batch in ds:
        if args.max_batches > 0 and batch_count >= args.max_batches:
            break

        obs = tf.convert_to_tensor(batch["obs"], dtype=tf.float32)
        mask = tf.convert_to_tensor(batch["mask"], dtype=tf.float32)
        map_state = batch_value_or_none(batch, "map_state")
        vision = batch_value_or_none(batch, "vision")

        out = student(
            obs,
            mask=mask,
            map_state=map_state,
            vision=vision,
            training=False,
            return_aux=True,
        )

        gate_probs = out["gate_probs"].numpy()               # [B, N]
        expert_id = np.argmax(gate_probs, axis=-1)          # [B]
        task_embedding = out["task_embedding"].numpy()      # [B, D]
        backbone_feat = out["backbone_feat"].numpy()        # [B, D]
        expert_outs = ensure_expert_tensor(out["expert_outs"]).numpy()  # [B, N, D]

        all_gate_probs.append(gate_probs)
        all_expert_ids.append(expert_id)
        all_task_emb.append(task_embedding)
        all_backbone.append(backbone_feat)
        all_expert_outs.append(expert_outs)

        # summary row per batch
        mean_gate = gate_probs.mean(axis=0)
        row = {"batch": batch_count}
        for i in range(gate_probs.shape[1]):
            row[f"mean_gate_expert_{i}"] = float(mean_gate[i])
            row[f"argmax_frac_expert_{i}"] = float((expert_id == i).mean())
        rows.append(row)

        batch_count += 1

    gate_probs_np = np.concatenate(all_gate_probs, axis=0)       # [M, N]
    expert_ids_np = np.concatenate(all_expert_ids, axis=0)       # [M]
    task_emb_np = np.concatenate(all_task_emb, axis=0)           # [M, D]
    backbone_np = np.concatenate(all_backbone, axis=0)           # [M, D]
    expert_outs_np = np.concatenate(all_expert_outs, axis=0)     # [M, N, D]

    mean_gate_global = gate_probs_np.mean(axis=0)
    argmax_usage = np.array([(expert_ids_np == i).mean() for i in range(gate_probs_np.shape[1])])

    sim_mat = cosine_similarity_matrix(expert_outs_np)

    summary = {"restored_ckpt": restored_ckpt,
               "num_samples": int(gate_probs_np.shape[0]),
               "num_experts": int(gate_probs_np.shape[1]),
               "mean_gate_probs": mean_gate_global.tolist(),
               "argmax_usage": argmax_usage.tolist(),
               "cosine_similarity_matrix": sim_mat.tolist(),
               "mean_offdiag_similarity": float(sim_mat[~np.eye(sim_mat.shape[0], dtype=bool)].mean())
                                                if sim_mat.shape[0] > 1 else 0.0,
              } 

    print("\n===== Expert analysis summary =====")
    print("mean_gate_probs:", mean_gate_global)
    print("argmax_usage   :", argmax_usage)
    print("similarity_mat :\n", sim_mat)

    save_csv(os.path.join(args.outdir, "batch_gate_summary.csv"), rows)

    with open(os.path.join(args.outdir, "expert_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    np.savez_compressed(os.path.join(args.outdir, "expert_analysis_outputs.npz"),
                        gate_probs=gate_probs_np,
                        expert_id=expert_ids_np,
                        task_embedding=task_emb_np,
                        backbone_feat=backbone_np,
                        expert_outs=expert_outs_np,
                       )

    # plots
    plot_heatmap(sim_mat, os.path.join(args.outdir, "expert_similarity_heatmap.png"),
                 "Mean cosine similarity between experts"
                 )

    # TSNE 1: expert outputs stacked and colored by expert index
    m, n, d = expert_outs_np.shape
    expert_outs_stacked = expert_outs_np.reshape(m * n, d)
    expert_labels_stacked = np.repeat(np.arange(n), m)

    X2_expert = maybe_run_tsne(expert_outs_stacked)
    if X2_expert is not None:plot_scatter_2d(X2_expert,
                                             expert_labels_stacked,
                                             os.path.join(args.outdir, "tsne_expert_outputs.png"),
                                             "t-SNE of expert outputs",
                                             "expert_",
                                            )
    else:
        print("sklearn not available: skipping t-SNE of expert outputs.")

    # TSNE 2: backbone features colored by dominant expert
    X2_backbone = maybe_run_tsne(backbone_np)
    if X2_backbone is not None:
        plot_scatter_2d(X2_backbone,
                        expert_ids_np,
                        os.path.join(args.outdir, "tsne_backbone_by_expert.png"),
                        "t-SNE of backbone features colored by dominant expert",
                        "expert_",
                      )
    else:
        print("sklearn not available: skipping t-SNE of backbone features.")

    print("\nSaved analysis to:", args.outdir)


if __name__ == "__main__":
    main()