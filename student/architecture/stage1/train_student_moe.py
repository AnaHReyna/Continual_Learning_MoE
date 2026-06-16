from email import parser
import os
import sys
sys.path.append('../../')

import csv
import json
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

from train.init_configs import get_argument, set_configs
from common.student_dataset import DistillTransitionDataset
from stage1.student_model_moe import StudentMoE
from common.moe_layers import router_balance_loss, router_entropy


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


def save_rows_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_rows_json(path, rows):
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)


def weighted_action_loss(pred, target, steer_weight=2.0, speed_weight=1.0):
    """
    action[..., 0] = speed_km
    action[..., 1] = steer
    """
    pred = tf.convert_to_tensor(pred, dtype=tf.float32)
    target = tf.convert_to_tensor(target, dtype=tf.float32)

    speed_pred = pred[..., 0]
    speed_tgt = target[..., 0]
    speed_mse = tf.reduce_mean(tf.square(speed_pred - speed_tgt))
    speed_mae = tf.reduce_mean(tf.abs(speed_pred - speed_tgt))

    steer_pred = pred[..., 1]
    steer_tgt = target[..., 1]
    steer_mse = tf.reduce_mean(tf.square(steer_pred - steer_tgt))
    steer_mae = tf.reduce_mean(tf.abs(steer_pred - steer_tgt))

    total_mse = speed_weight * speed_mse + steer_weight * steer_mse
    total_mae = speed_weight * speed_mae + steer_weight * steer_mae

    return total_mse, total_mae, speed_mse, steer_mse



def expert_diversity_loss(expert_outs):
    """
    expert_outs:
      - list/tuple de [B, D], um por expert
      - ou tensor [B, N, D]
    """
    if isinstance(expert_outs, (list, tuple)):
        expert_outs = tf.stack(expert_outs, axis=1)   # [B, N, D]

    expert_outs = tf.cast(expert_outs, tf.float32)
    expert_outs = tf.math.l2_normalize(expert_outs, axis=-1)

    # similaridade par-a-par: [B, N, N]
    sim = tf.einsum("bid,bjd->bij", expert_outs, expert_outs)

    n = tf.shape(sim)[-1]
    eye = tf.eye(n, dtype=tf.float32)
    pair_mask = 1.0 - eye

    # penaliza experts muito parecidos
    sim2 = tf.square(sim)

    num = tf.reduce_sum(sim2 * pair_mask[None, :, :])
    den = tf.reduce_sum(pair_mask) * tf.cast(tf.shape(sim)[0], tf.float32) + 1e-8

    return num / den


def get_diversity_weight(epoch, max_weight, warmup_epochs, ramp_epochs):
    """
    epoch começa em 1
    """
    if max_weight <= 0.0:
        return 0.0

    if epoch <= warmup_epochs:
        return 0.0

    if ramp_epochs <= 0:
        return float(max_weight)

    progress = (epoch - warmup_epochs) / float(ramp_epochs)
    progress = max(0.0, min(1.0, progress))
    return float(max_weight) * progress


def supervised_contrastive_loss(embeddings, labels, temperature=0.1):
    z = tf.math.l2_normalize(tf.cast(embeddings, tf.float32), axis=1)
    labels = tf.cast(labels, tf.int32)

    logits = tf.matmul(z, z, transpose_b=True) / temperature
    logits = logits - tf.reduce_max(logits, axis=1, keepdims=True)

    batch_size = tf.shape(z)[0]
    eye = tf.eye(batch_size, dtype=tf.bool)

    same = tf.equal(labels[:, None], labels[None, :])
    pos_mask = tf.logical_and(same, tf.logical_not(eye))
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



def task_alignment_loss(student_task_embedding, teacher_task_embedding):
    z_s = tf.math.l2_normalize(tf.cast(student_task_embedding, tf.float32), axis=-1)
    z_t = tf.math.l2_normalize(tf.cast(teacher_task_embedding, tf.float32), axis=-1)
    cos_sim = tf.reduce_sum(z_s * z_t, axis=-1)
    return tf.reduce_mean(1.0 - cos_sim)


def main():
    parser = get_argument()
    parser.add_argument("--dataset-dirs", nargs="+", required=True)
    parser.add_argument("--student-logdir", type=str, default="checkpoints")
    parser.add_argument("--student-name", type=str, default="stage1_moe_lane")
    parser.add_argument("--student-epochs", type=int, default=150)
    parser.add_argument("--student-batch-size", type=int, default=64)
    parser.add_argument("--student-lr", type=float, default=1e-4)
    parser.add_argument("--student-target-mode", choices=["mean_action", "raw_mean"], default="mean_action")
    parser.add_argument("--student-use-vision", action="store_true")
    parser.add_argument("--student-vision-dim", type=int, default=280)
    parser.add_argument("--student-fusion-type", choices=["cross", "self"], default="cross")
    parser.add_argument("--student-save-every", type=int, default=5)

    parser.add_argument("--num-experts", type=int, default=2)
    parser.add_argument("--task-dim", type=int, default=16)
    parser.add_argument("--router-balance-weight", type=float, default=0.01)
    parser.add_argument("--router-entropy-weight", type=float, default=0.0)

    parser.add_argument("--steer-weight", type=float, default=2.0)
    parser.add_argument("--speed-weight", type=float, default=1.0)

    parser.add_argument("--expert-diversity-weight", type=float, default=0.0)
    parser.add_argument("--expert-diversity-warmup-epochs", type=int, default=5)
    parser.add_argument("--expert-diversity-ramp-epochs", type=int, default=10)

    parser.add_argument("--teacher-task-prototypes", type=str, default=None)

    parser.add_argument("--use-task-contrastive", action="store_true")
    parser.add_argument("--task-contrastive-weight", type=float, default=0.0)
    parser.add_argument("--task-contrastive-temp", type=float, default=0.1)

    parser.add_argument("--use-task-alignment", action="store_true")
    parser.add_argument("--task-alignment-weight", type=float, default=0.0)

    args = parser.parse_args()
    args, algo_params, _ = set_configs(args, test=False)

    gpus = tf.config.experimental.list_physical_devices("GPU")
    if args.gpu is not None and args.gpu >= 0 and len(gpus) > args.gpu:
        tf.config.set_visible_devices([gpus[args.gpu]], "GPU")
        tf.config.experimental.set_memory_growth(gpus[args.gpu], True)
        print("Using GPU:", gpus[args.gpu])
    else:
        tf.config.set_visible_devices([], "GPU")
        print("Training on CPU.")

    dataset = DistillTransitionDataset(dataset_dirs=args.dataset_dirs,
                                       target_mode=args.student_target_mode,
                                       use_vision=args.student_use_vision,
                                       require_map=True,
                                       teacher_task_prototypes_path=args.teacher_task_prototypes,
                                      )

    train_ds = dataset.make_tf_dataset(batch_size=args.student_batch_size,
                                       shuffle=True,
                                       shuffle_buffer=10000,
                                       repeat=False,
                                       )

    sample0 = dataset.get_specimen()
    state_shape = sample0["obs"].shape
    action_dim = sample0["act"].shape[-1]

    student_params = build_student_params(args, algo_params)

    policy = StudentMoE(params=student_params,
                        state_shape=state_shape,
                        action_dim=action_dim,
                        max_action=1.0,
                        num_experts=args.num_experts,
                        task_dim=args.task_dim,
                        )
    
    policy.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.student_lr)

    ckpt = tf.train.Checkpoint(student=policy, optimizer=optimizer)
    ckpt_dir = os.path.join(args.student_logdir, args.student_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=50)

    tb_train = tf.summary.create_file_writer(os.path.join(ckpt_dir, "tb", "train"))

    metrics = []
    start_dt = datetime.now()
    start_perf = time.perf_counter()

    with open(os.path.join(ckpt_dir, "run_config.json"), "w") as f:
        json.dump({"dataset_dirs": args.dataset_dirs,
                   "student_target_mode": args.student_target_mode,
                   "student_use_vision": args.student_use_vision,
                   "student_vision_dim": args.student_vision_dim,
                   "student_fusion_type": args.student_fusion_type,
                   "num_experts": args.num_experts,
                   "task_dim": args.task_dim,
                   "router_balance_weight": args.router_balance_weight,
                   "router_entropy_weight": args.router_entropy_weight,
                   "start_time": start_dt.isoformat(),
                   "task_mapping": dataset.task_to_id,
                   "expert_diversity_weight": args.expert_diversity_weight,
                   "expert_diversity_warmup_epochs": args.expert_diversity_warmup_epochs,
                   "expert_diversity_ramp_epochs": args.expert_diversity_ramp_epochs,
                   "teacher_task_prototypes": args.teacher_task_prototypes,
                   "use_task_contrastive": args.use_task_contrastive,
                   "task_contrastive_weight": args.task_contrastive_weight,
                   "task_contrastive_temp": args.task_contrastive_temp,
                   "use_task_alignment": args.use_task_alignment,
                   "task_alignment_weight": args.task_alignment_weight,
                   }, 
                   f, 
                   indent=2
                  )
        

    @tf.function
    def train_step(batch, current_div_weight):
        obs = tf.convert_to_tensor(batch["obs"], dtype=tf.float32)
        mask = tf.convert_to_tensor(batch["mask"], dtype=tf.float32)
        map_state = batch_value_or_none(batch, "map_state")
        vision = batch_value_or_none(batch, "vision")
        target = tf.convert_to_tensor(batch["target"], dtype=tf.float32)

        task_id = tf.convert_to_tensor(batch["task_id"], dtype=tf.int32)
        teacher_task_embedding = batch_value_or_none(batch, "teacher_task_embedding")

        with tf.GradientTape() as tape:
            out = policy(obs,
                         mask=mask,
                         map_state=map_state,
                         vision=vision,
                         training=True,
                         return_aux=True,
                        )

            pred = out["raw_mean"] if args.student_target_mode == "raw_mean" else out["action"]

            distill_loss, mae, speed_mse, steer_mse = weighted_action_loss(pred, 
                                                                           target, 
                                                                           steer_weight=args.steer_weight, 
                                                                           speed_weight=args.speed_weight,
                                                                           )

            balance_loss = router_balance_loss(out["gate_probs"])
            ent = router_entropy(out["gate_probs"])

            div_loss = tf.constant(0.0, dtype=tf.float32)

            task_ctr_loss = tf.constant(0.0, dtype=tf.float32)
            if args.use_task_contrastive and args.task_contrastive_weight > 0.0:
                task_ctr_loss = supervised_contrastive_loss(out["task_embedding"], 
                                                            task_id,
                                                            temperature=args.task_contrastive_temp,
                                                            )

            task_align_loss = tf.constant(0.0, dtype=tf.float32)
            if (args.use_task_alignment
                and args.task_alignment_weight > 0.0
                and teacher_task_embedding is not None
               ):
                
                task_align_loss = task_alignment_loss(out["task_embedding"],
                                                      teacher_task_embedding,
                                                      )

            if current_div_weight > 0.0:
                div_loss = expert_diversity_loss(out["expert_outs"])

            total_loss = distill_loss
            total_loss += args.router_balance_weight * balance_loss
            total_loss += args.router_entropy_weight * ent
            total_loss += tf.cast(current_div_weight, tf.float32) * div_loss
            total_loss += args.task_contrastive_weight * task_ctr_loss
            total_loss += args.task_alignment_weight * task_align_loss

        grads = tape.gradient(total_loss, policy.trainable_variables)
        optimizer.apply_gradients(zip(grads, policy.trainable_variables))

        return (total_loss,
                distill_loss,
                mae,
                balance_loss,
                ent,
                div_loss,
                task_ctr_loss,
                task_align_loss,
                speed_mse,
                steer_mse,
            )
    


    for epoch in range(1, args.student_epochs + 1):
        epoch_start = time.perf_counter()
        total_losses = []
        distill_losses = []
        maes = []
        balances = []
        ents = []
        div_losses = []

        task_ctr_losses = []
        task_align_losses = []
        speed_mses = []
        steer_mses = []

        current_div_weight = get_diversity_weight(
            epoch=epoch,
            max_weight=args.expert_diversity_weight,
            warmup_epochs=args.expert_diversity_warmup_epochs,
            ramp_epochs=args.expert_diversity_ramp_epochs,
        )

        for batch in train_ds:
            (
                total_loss,
                distill_loss,
                mae,
                balance_loss,
                ent,
                div_loss,
                task_ctr_loss,
                task_align_loss,
                speed_mse,
                steer_mse,
            ) = train_step(
                batch,
                tf.constant(current_div_weight, dtype=tf.float32),
            )

            total_losses.append(float(total_loss.numpy()))
            distill_losses.append(float(distill_loss.numpy()))
            maes.append(float(mae.numpy()))
            balances.append(float(balance_loss.numpy()))
            ents.append(float(ent.numpy()))
            div_losses.append(float(div_loss.numpy()))
            task_ctr_losses.append(float(task_ctr_loss.numpy()))
            task_align_losses.append(float(task_align_loss.numpy()))
            speed_mses.append(float(speed_mse.numpy()))
            steer_mses.append(float(steer_mse.numpy()))


        row = {
            "epoch": epoch,
            "total_loss": float(np.mean(total_losses)),
            "distill_mse": float(np.mean(distill_losses)),
            "action_mae": float(np.mean(maes)),
            "router_balance_loss": float(np.mean(balances)),
            "router_entropy": float(np.mean(ents)),
            "epoch_duration_sec": float(time.perf_counter() - epoch_start),
            "speed_mse": float(np.mean(speed_mses)),
            "steer_mse": float(np.mean(steer_mses)),
            "expert_diversity_loss": float(np.mean(div_losses)),
            "expert_diversity_weight": float(current_div_weight),
            "task_contrastive_loss": float(np.mean(task_ctr_losses)),
            "task_alignment_loss": float(np.mean(task_align_losses)),
        }
        
        metrics.append(row)

        with tb_train.as_default():
            tf.summary.scalar("train/total_loss", row["total_loss"], step=epoch)
            tf.summary.scalar("train/distill_mse", row["distill_mse"], step=epoch)
            tf.summary.scalar("train/action_mae", row["action_mae"], step=epoch)
            tf.summary.scalar("train/router_balance_loss", row["router_balance_loss"], step=epoch)
            tf.summary.scalar("train/router_entropy", row["router_entropy"], step=epoch)
            tf.summary.scalar("train/epoch_duration_sec", row["epoch_duration_sec"], step=epoch)
            tf.summary.scalar("train/steer_mse", row["steer_mse"], step=epoch)
            tf.summary.scalar("train/expert_diversity_loss", row["expert_diversity_loss"], step=epoch)
            tf.summary.scalar("train/expert_diversity_weight", row["expert_diversity_weight"], step=epoch)
            tf.summary.scalar("train/speed_mse", row["speed_mse"], step=epoch)
            tf.summary.scalar("train/task_contrastive_loss", row["task_contrastive_loss"], step=epoch)
            tf.summary.scalar("train/task_alignment_loss", row["task_alignment_loss"], step=epoch)
        tb_train.flush()

        print(f"[StudentMoE] epoch={epoch:03d} | "
             f"total_loss={row['total_loss']:.6f} | "
             f"distill_mse={row['distill_mse']:.6f} | "
             f"speed_mse={row['speed_mse']:.6f} | "
             f"steer_mse={row['steer_mse']:.6f} | "
             f"action_mae={row['action_mae']:.6f} | "
             f"router_balance={row['router_balance_loss']:.6f} | "
             f"div={row['expert_diversity_loss']:.6f} | "
             f"div_w={row['expert_diversity_weight']:.6f} | "
             f"task_ctr={row['task_contrastive_loss']:.6f} | "
             f"task_align={row['task_alignment_loss']:.6f}"
            )

        save_rows_json(os.path.join(ckpt_dir, "train_metrics.json"), metrics)
        save_rows_csv(os.path.join(ckpt_dir, "train_metrics.csv"), metrics)

        if epoch % args.student_save_every == 0 or epoch == args.student_epochs:
            path = manager.save()
            print(f"Saved StudentMoE checkpoint: {path}")

    end_dt = datetime.now()
    with open(os.path.join(ckpt_dir, "run_summary.json"), "w") as f:
        json.dump({"train_start_time": start_dt.isoformat(),
                   "train_end_time": end_dt.isoformat(),
                   "total_train_duration_sec": float(time.perf_counter() - start_perf),
                   "best_distill_mse": float(np.min([m["distill_mse"] for m in metrics])) if metrics else None,
                   "last_distill_mse": metrics[-1]["distill_mse"] if metrics else None,
                   "last_action_mae": metrics[-1]["action_mae"] if metrics else None,
                   }, 
                   f, 
                   indent=2
                   )


if __name__ == "__main__":
    main()


# python train_student_moe.py \
#   --dataset-dirs ../export_rollouts/datasets/lane_keeping \
#   --student-logdir checkpoints \
#   --student-name stage1_moe_lane_steer2 \
#   --student-epochs 60 \
#   --student-batch-size 64 \
#   --student-lr 1e-4 \
#   --student-target-mode mean_action \
#   --num-experts 2 \
#   --task-dim 16 \
#   --router-balance-weight 0.01
#   --steer-weight 2.0 \
#   --speed-weight 1.0
#######################################################################
# python train_student_moe.py \
#   --dataset-dirs ../../export_rollouts/datasets/lane_keeping \
#   --student-logdir checkpoints_div \
#   --student-name stage1_moe_lane_div \
#   --student-epochs 100 \
#   --student-batch-size 64 \
#   --student-lr 1e-4 \
#   --student-target-mode mean_action \
#   --num-experts 2 \
#   --task-dim 16 \
#   --router-balance-weight 0.01 \
#   --steer-weight 2.0 \
#   --speed-weight 1.0 \
#   --expert-diversity-weight 0.0005 \
#   --expert-diversity-warmup-epochs 5 \
#   --expert-diversity-ramp-epochs 10
######################################################################################
# python train_student_moe.py \
#   --dataset-dirs ../../export_rollouts/datasets/lane_keeping_eval ../../export_rollouts/datasets/change_lane_eval \
#   --teacher-task-prototypes teacher_task_prototypes/teacher_task_prototypes.npz \
#   --student-logdir checkpoints_stage1_multitask \
#   --student-name stage1_moe_lane_change \
#   --student-epochs 200 \
#   --student-batch-size 64 \
#   --student-lr 1e-4 \
#   --student-target-mode mean_action \
#   --num-experts 2 \
#   --task-dim 16 \
#   --router-balance-weight 0.01 \
#   --steer-weight 2.0 \
#   --speed-weight 1.0 \
#   --expert-diversity-weight 0.0005 \
#   --expert-diversity-warmup-epochs 5 \
#   --expert-diversity-ramp-epochs 10 \
#   --use-task-contrastive \
#   --task-contrastive-weight 0.01 \
#   --task-contrastive-temp 0.1 \
#   --use-task-alignment \
#   --task-alignment-weight 0.01
####################################################################################################
# python train_student_moe.py \
#   --dataset-dirs ../../export_rollouts/datasets/lane_keeping ../../export_rollouts/datasets/change_lane_eval \
#   --teacher-task-prototypes teacher_task_prototypes/teacher_task_prototypes.npz \
#   --student-logdir checkpoints_stage1_multitask \
#   --student-name stage1_moe_lane_change \
#   --student-epochs 200 \
#   --student-batch-size 64 \
#   --student-lr 1e-4 \
#   --student-save-every 10 \
#   --student-target-mode mean_action \
#   --num-experts 2 \
#   --task-dim 16 \
#   --router-balance-weight 0.01 \
#   --steer-weight 2.0 \
#   --speed-weight 1.0 \
#   --expert-diversity-weight 0.0005 \
#   --expert-diversity-warmup-epochs 5 \
#   --expert-diversity-ramp-epochs 10 \
#   --use-task-contrastive \
#   --task-contrastive-weight 0.01 \
#   --task-contrastive-temp 0.1 \
#   --use-task-alignment \
#   --task-alignment-weight 0.01