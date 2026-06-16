import os
import json
import argparse

import numpy as np
import tensorflow as tf

from prototypes.teacher_task_window_dataset import TeacherTaskWindowDataset
from prototypes.teacher_task_encoder import TeacherTaskEncoder


def l2_normalize_np(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (n + eps)


def restore_model(model, ckpt_dir, ckpt_id=None):
    ckpt = tf.train.Checkpoint(model=model)

    if ckpt_id is None or ckpt_id == "latest":
        ckpt_path = tf.train.latest_checkpoint(ckpt_dir)
        if ckpt_path is None:
            raise ValueError(f"No checkpoint found in: {ckpt_dir}")
    else:
        ckpt_path = os.path.join(ckpt_dir, ckpt_id)

    ckpt.restore(ckpt_path).expect_partial()
    print(f"Restored checkpoint: {ckpt_path}")
    return ckpt_path


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def build_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-dirs", nargs="+", required=True)
    parser.add_argument("--ckpt-dir", type=str, required=True)
    parser.add_argument("--ckpt-id", type=str, default="latest")
    parser.add_argument("--outdir", type=str, default="teacher_task_prototypes")

    parser.add_argument("--window-len", type=int, default=20)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)

    parser.add_argument("--action-source", type=str, default="teacher_mean_action",
                        choices=["teacher_mean_action", "act"])
    parser.add_argument("--reward-key", type=str, default="rew")
    parser.add_argument("--balance-tasks", action="store_true")

    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--task-emb-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use-cls-token", action="store_true")

    parser.add_argument("--save-window-embeddings", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)

    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    gpus = tf.config.experimental.list_physical_devices("GPU")
    if args.gpu is not None and args.gpu >= 0 and len(gpus) > args.gpu:
        tf.config.set_visible_devices([gpus[args.gpu]], "GPU")
        tf.config.experimental.set_memory_growth(gpus[args.gpu], True)
        print("Using GPU:", gpus[args.gpu])
    else:
        tf.config.set_visible_devices([], "GPU")
        print("Running on CPU.")

    os.makedirs(args.outdir, exist_ok=True)

    dataset = TeacherTaskWindowDataset(
        dataset_dirs=args.dataset_dirs,
        window_len=args.window_len,
        stride=args.stride,
        action_source=args.action_source,
        reward_key=args.reward_key,
        balance_tasks=args.balance_tasks,
        pad_short_windows=True,
    )

    print("Task counts:", dataset.get_task_counts())

    ds = dataset.make_tf_dataset(
        batch_size=args.batch_size,
        shuffle=False,
        shuffle_buffer=1,
        repeat=False,
    )

    specimen = dataset.get_specimen()
    state_dim = specimen["states"].shape[-1]
    action_dim = specimen["actions"].shape[-1]

    model = TeacherTaskEncoder(
        state_dim=state_dim,
        action_dim=action_dim,
        window_len=args.window_len,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        num_layers=args.num_layers,
        task_emb_dim=args.task_emb_dim,
        dropout=args.dropout,
        use_cls_token=args.use_cls_token,
    )

    restored_ckpt = restore_model(model, args.ckpt_dir, args.ckpt_id)

    all_embeddings = []
    all_task_ids = []

    for batch in ds:
        states = tf.convert_to_tensor(batch["states"], dtype=tf.float32)
        actions = tf.convert_to_tensor(batch["actions"], dtype=tf.float32)
        rewards = tf.convert_to_tensor(batch["rewards"], dtype=tf.float32)
        mask = tf.convert_to_tensor(batch["mask"], dtype=tf.float32)
        task_ids = tf.convert_to_tensor(batch["task_id"], dtype=tf.int32)

        z = model(
            states=states,
            actions=actions,
            rewards=rewards,
            mask=mask,
            training=False,
        )

        all_embeddings.append(z.numpy())
        all_task_ids.append(task_ids.numpy())

    embeddings_np = np.concatenate(all_embeddings, axis=0).astype(np.float32)   # [N, D]
    task_ids_np = np.concatenate(all_task_ids, axis=0).astype(np.int32)          # [N]

    inv_task_map = {v: k for k, v in dataset.task_to_id.items()}
    sorted_task_ids = sorted(inv_task_map.keys())

    prototypes = []
    task_names = []
    counts = []

    for tid in sorted_task_ids:
        idx = task_ids_np == tid
        z_task = embeddings_np[idx]
        proto = z_task.mean(axis=0, keepdims=False)
        proto = l2_normalize_np(proto)

        prototypes.append(proto.astype(np.float32))
        task_names.append(inv_task_map[tid])
        counts.append(int(idx.sum()))

    prototypes_np = np.stack(prototypes, axis=0).astype(np.float32)   # [K, D]
    task_ids_arr = np.asarray(sorted_task_ids, dtype=np.int32)
    task_names_arr = np.asarray(task_names)

    np.savez_compressed(
        os.path.join(args.outdir, "teacher_task_prototypes.npz"),
        prototypes=prototypes_np,
        task_ids=task_ids_arr,
        task_names=task_names_arr,
        counts=np.asarray(counts, dtype=np.int32),
        restored_ckpt=np.asarray([restored_ckpt]),
    )

    summary = {
        "restored_ckpt": restored_ckpt,
        "num_windows": int(embeddings_np.shape[0]),
        "task_ids": task_ids_arr.tolist(),
        "task_names": task_names,
        "counts": counts,
        "prototype_shape": list(prototypes_np.shape),
    }
    save_json(os.path.join(args.outdir, "prototype_summary.json"), summary)

    if args.save_window_embeddings:
        np.savez_compressed(
            os.path.join(args.outdir, "teacher_window_embeddings.npz"),
            embeddings=embeddings_np,
            task_ids=task_ids_np,
        )

    print("\nSaved prototypes to:", args.outdir)
    print("Task names:", task_names)
    print("Prototype shape:", prototypes_np.shape)


if __name__ == "__main__":
    main()

# python build_teacher_task_prototypes.py \
#   --dataset-dirs ../../export_rollouts/datasets/lane_keeping ../../export_rollouts/datasets/change_lane_eval \
#   --ckpt-dir teacher_task_encoder_ckpts \
#   --ckpt-id ckpt-20 \
#   --outdir teacher_task_prototypes \
#   --window-len 20 \
#   --stride 10 \
#   --batch-size 64 \
#   --action-source teacher_mean_action \
#   --reward-key rew \
#   --model-dim 128 \
#   --num-heads 4 \
#   --ff-dim 256 \
#   --num-layers 2 \
#   --task-emb-dim 16 \
#   --dropout 0.1 \
#   --use-cls-token \
#   --save-window-embeddings

###########################################################################################################################
# python build_teacher_task_prototypes.py \
#   --dataset-dirs ../../export_rollouts/datasets/lane_keeping ../../export_rollouts/datasets/change_lane_eval ../../export_rollouts/datasets/pedestrian \
#   --ckpt-dir teacher_task_encoder_ckpts_stage2 \
#   --ckpt-id ckpt-20 \
#   --outdir teacher_task_prototypes_stage2 \
#   --window-len 20 \
#   --stride 10 \
#   --batch-size 64 \
#   --action-source teacher_mean_action \
#   --reward-key rew \
#   --model-dim 128 \
#   --num-heads 4 \
#   --ff-dim 256 \
#   --num-layers 2 \
#   --task-emb-dim 16 \
#   --dropout 0.1 \
#   --use-cls-token \
#   --save-window-embeddings