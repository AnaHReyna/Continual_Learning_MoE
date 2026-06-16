import os
import json
import time
import argparse
from datetime import datetime

import numpy as np
import tensorflow as tf

from prototypes.teacher_task_window_dataset import TeacherTaskWindowDataset
from prototypes.teacher_task_encoder import TeacherTaskEncoder


def supervised_infonce_loss(embeddings, task_ids, temperature=0.1):
    """
    embeddings: [B, D]
    task_ids   : [B]
    """
    z = tf.math.l2_normalize(tf.cast(embeddings, tf.float32), axis=1)
    task_ids = tf.cast(task_ids, tf.int32)

    logits = tf.matmul(z, z, transpose_b=True) / temperature
    logits = logits - tf.reduce_max(logits, axis=1, keepdims=True)

    batch_size = tf.shape(z)[0]
    eye = tf.eye(batch_size, dtype=tf.bool)

    same_task = tf.equal(task_ids[:, None], task_ids[None, :])
    pos_mask = tf.logical_and(same_task, tf.logical_not(eye))
    valid_mask = tf.logical_not(eye)

    exp_logits = tf.exp(logits) * tf.cast(valid_mask, tf.float32)
    log_prob = logits - tf.math.log(tf.reduce_sum(exp_logits, axis=1, keepdims=True) + 1e-8)

    pos_mask_f = tf.cast(pos_mask, tf.float32)
    pos_count = tf.reduce_sum(pos_mask_f, axis=1)

    mean_log_prob_pos = tf.reduce_sum(pos_mask_f * log_prob, axis=1) / tf.maximum(pos_count, 1.0)

    valid_rows = pos_count > 0.0
    loss_vec = -mean_log_prob_pos
    loss_vec = tf.boolean_mask(loss_vec, valid_rows)

    return tf.cond(
        tf.size(loss_vec) > 0,
        lambda: tf.reduce_mean(loss_vec),
        lambda: tf.constant(0.0, dtype=tf.float32),
    )


def save_rows_json(path, rows):
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)


def build_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-dirs", nargs="+", required=True)
    parser.add_argument("--outdir", type=str, default="teacher_task_encoder_ckpts")

    parser.add_argument("--window-len", type=int, default=20)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save-every", type=int, default=5)

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
    parser.add_argument("--temperature", type=float, default=0.1)

    parser.add_argument("--use-cls-token", action="store_true")

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

    dataset = TeacherTaskWindowDataset(dataset_dirs=args.dataset_dirs,
                                       window_len=args.window_len,
                                       stride=args.stride,
                                       action_source=args.action_source,
                                       reward_key=args.reward_key,
                                       balance_tasks=args.balance_tasks,
                                       pad_short_windows=True,
                                       )

    print("Task counts:", dataset.get_task_counts())

    ds = dataset.make_tf_dataset(batch_size=args.batch_size,
                                 shuffle=True,
                                 shuffle_buffer=10000,
                                 repeat=False,
                                )

    specimen = dataset.get_specimen()
    state_dim = specimen["states"].shape[-1]
    action_dim = specimen["actions"].shape[-1]

    model = TeacherTaskEncoder(state_dim=state_dim,
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

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    manager = tf.train.CheckpointManager(ckpt, args.outdir, max_to_keep=20)

    metrics = []
    start_dt = datetime.now()
    start_perf = time.perf_counter()

    with open(os.path.join(args.outdir, "run_config.json"), "w") as f:
        json.dump({
            "dataset_dirs": args.dataset_dirs,
            "window_len": args.window_len,
            "stride": args.stride,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "action_source": args.action_source,
            "reward_key": args.reward_key,
            "balance_tasks": args.balance_tasks,
            "model_dim": args.model_dim,
            "num_heads": args.num_heads,
            "ff_dim": args.ff_dim,
            "num_layers": args.num_layers,
            "task_emb_dim": args.task_emb_dim,
            "dropout": args.dropout,
            "temperature": args.temperature,
            "use_cls_token": args.use_cls_token,
            "task_mapping": dataset.task_to_id,
            "start_time": start_dt.isoformat(),
        }, f, indent=2)

    @tf.function
    def train_step(batch):
        states = tf.convert_to_tensor(batch["states"], dtype=tf.float32)
        actions = tf.convert_to_tensor(batch["actions"], dtype=tf.float32)
        rewards = tf.convert_to_tensor(batch["rewards"], dtype=tf.float32)
        mask = tf.convert_to_tensor(batch["mask"], dtype=tf.float32)
        task_ids = tf.convert_to_tensor(batch["task_id"], dtype=tf.int32)

        with tf.GradientTape() as tape:
            z = model(
                states=states,
                actions=actions,
                rewards=rewards,
                mask=mask,
                training=True,
            )
            loss = supervised_infonce_loss(
                embeddings=z,
                task_ids=task_ids,
                temperature=args.temperature,
            )

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return loss, z

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        losses = []
        emb_norms = []

        for batch in ds:
            loss, z = train_step(batch)
            losses.append(float(loss.numpy()))
            emb_norms.append(float(tf.reduce_mean(tf.norm(z, axis=-1)).numpy()))

        row = {
            "epoch": epoch,
            "infonce_loss": float(np.mean(losses)) if losses else 0.0,
            "mean_embedding_norm": float(np.mean(emb_norms)) if emb_norms else 0.0,
            "epoch_duration_sec": float(time.perf_counter() - epoch_start),
        }
        metrics.append(row)

        print(
            f"[TeacherTaskEncoder] epoch={epoch:03d} | "
            f"infonce_loss={row['infonce_loss']:.6f} | "
            f"mean_emb_norm={row['mean_embedding_norm']:.6f}"
        )

        save_rows_json(os.path.join(args.outdir, "train_metrics.json"), metrics)

        if epoch % args.save_every == 0 or epoch == args.epochs:
            path = manager.save()
            print(f"Saved checkpoint: {path}")

    end_dt = datetime.now()
    with open(os.path.join(args.outdir, "run_summary.json"), "w") as f:
        json.dump({
            "train_start_time": start_dt.isoformat(),
            "train_end_time": end_dt.isoformat(),
            "total_train_duration_sec": float(time.perf_counter() - start_perf),
            "best_infonce_loss": float(np.min([m["infonce_loss"] for m in metrics])) if metrics else None,
            "last_infonce_loss": metrics[-1]["infonce_loss"] if metrics else None,
        }, f, indent=2)


if __name__ == "__main__":
    main()


# python train_teacher_task_encoder.py \
#   --dataset-dirs ../../export_rollouts/datasets/lane_keeping ../../export_rollouts/datasets/change_lane_eval \
#   --outdir teacher_task_encoder_ckpts \
#   --window-len 20 \
#   --stride 10 \
#   --batch-size 64 \
#   --epochs 100 \
#   --lr 1e-4 \
#   --action-source teacher_mean_action \
#   --reward-key rew \
#   --balance-tasks \
#   --model-dim 128 \
#   --num-heads 4 \
#   --ff-dim 256 \
#   --num-layers 2 \
#   --task-emb-dim 16 \
#   --dropout 0.1 \
#   --temperature 0.1 \
#   --use-cls-token

##################################################################################

# python train_teacher_task_encoder.py \
#   --dataset-dirs ../../export_rollouts/datasets/lane_keeping ../../export_rollouts/datasets/change_lane_eval ../../export_rollouts/datasets/pedestrian \
#   --outdir teacher_task_encoder_ckpts_stage2 \
#   --window-len 20 \
#   --stride 10 \
#   --batch-size 64 \
#   --epochs 100 \
#   --lr 1e-4 \
#   --action-source teacher_mean_action \
#   --reward-key rew \
#   --balance-tasks \
#   --model-dim 128 \
#   --num-heads 4 \
#   --ff-dim 256 \
#   --num-layers 2 \
#   --task-emb-dim 16 \
#   --dropout 0.1 \
#   --temperature 0.1 \
#   --use-cls-token