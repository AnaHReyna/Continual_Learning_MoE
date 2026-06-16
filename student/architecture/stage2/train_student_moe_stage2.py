from html import parser
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
from stage2.student_model_moe_stage2 import StudentMoEStage2
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


def build_stage1_temp_model(params, state_shape, action_dim, num_old_experts, task_dim):
    return StudentMoE(params=params,
                      state_shape=state_shape,
                      action_dim=action_dim,
                      max_action=1.0,
                      num_experts=num_old_experts,
                      task_dim=task_dim,
                     )


def restore_stage1_moe(model, ckpt_path):
    ckpt = tf.train.Checkpoint(student=model)
    if tf.io.gfile.isdir(ckpt_path):
        latest = tf.train.latest_checkpoint(ckpt_path)
        if latest is None:
            raise ValueError(f"No checkpoint found inside directory: {ckpt_path}")
        ckpt_path = latest
    ckpt.restore(ckpt_path).expect_partial()
    print(f"[Stage2MoE] Restored stage1 MoE checkpoint: {ckpt_path}")
    return ckpt_path



def copy_weights_stage1_to_stage2(stage1_model, stage2_model):
    # backbone compartilhado
    for v_src, v_dst in zip(stage1_model.encoder.weights, stage2_model.encoder.weights):
        if v_src.shape == v_dst.shape:
            v_dst.assign(v_src)

    # task encoder
    for v_src, v_dst in zip(stage1_model.task_encoder.weights, stage2_model.task_encoder.weights):
        if v_src.shape == v_dst.shape:
            v_dst.assign(v_src)

    # experts antigos
    for i in range(stage2_model.num_old_experts):
        src_expert = stage1_model.moe.experts[i]
        dst_expert = stage2_model.moe.experts[i]

        for v_src, v_dst in zip(src_expert.weights, dst_expert.weights):
            if v_src.shape == v_dst.shape:
                v_dst.assign(v_src)

    # pre_router
    old_w, old_b = stage1_model.pre_router.get_weights()
    new_w, new_b = stage2_model.pre_router.get_weights()

    rows = min(old_w.shape[0], new_w.shape[0])
    new_w[:rows, :] = old_w[:rows, :]
    new_b[:] = old_b
    if new_w.shape[0] > old_w.shape[0]:
        new_w[old_w.shape[0]:, :] = 0.0

    stage2_model.pre_router.set_weights([new_w, new_b])

    # router: copia a parte antiga e zera as colunas do expert novo
    old_w, old_b = stage1_model.router.logits.get_weights()
    new_w, new_b = stage2_model.router.logits.get_weights()

    new_w[:, :stage2_model.num_old_experts] = old_w
    new_b[:stage2_model.num_old_experts] = old_b

    if stage2_model.num_total_experts > stage2_model.num_old_experts:
        new_w[:, stage2_model.num_old_experts:] = 0.0
        new_b[stage2_model.num_old_experts:] = 0.0

    stage2_model.router.logits.set_weights([new_w, new_b])

    # output head
    for v_src, v_dst in zip(stage1_model.out_mean.weights, stage2_model.out_mean.weights):
        if v_src.shape == v_dst.shape:
            v_dst.assign(v_src)


def initialize_new_experts_from_old_average(stage2_model, noise_std=1e-3):
    """
    Inicializa os experts novos como média dos experts antigos,
    com pequeno ruído opcional para evitar cópia idêntica.
    """
    num_old = stage2_model.num_old_experts
    num_total = stage2_model.num_total_experts

    if num_total <= num_old:
        return

    old_experts = stage2_model.moe.experts[:num_old]
    new_experts = stage2_model.moe.experts[num_old:]

    for new_expert in new_experts:
        for weight_idx, new_var in enumerate(new_expert.weights):
            old_vars = [old_expert.weights[weight_idx] for old_expert in old_experts]

            # Só faz média se os shapes forem iguais
            if all(old_var.shape == new_var.shape for old_var in old_vars):
                avg_value = tf.add_n([tf.cast(v, tf.float32) for v in old_vars]) / float(num_old)

                if noise_std > 0.0:
                    noise = tf.random.normal(shape=tf.shape(avg_value), stddev=noise_std)
                    avg_value = avg_value + noise

                new_var.assign(avg_value)

    print(
        f"[Stage2MoE] Initialized {len(new_experts)} new expert(s) "
        f"from the average of {num_old} old experts with noise_std={noise_std}"
    )
            

def weighted_action_loss(pred, target, speed_weight=1.0, steer_weight=2.0):
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


def supervised_contrastive_loss(embeddings, labels, temperature=0.1):
    """
    embeddings: [B, D]
    labels: [B] with integer labels, e.g. 0=straight, 1=curve
    """
    embeddings = tf.cast(embeddings, tf.float32)
    embeddings = tf.math.l2_normalize(embeddings, axis=1)

    labels = tf.reshape(tf.cast(labels, tf.int32), [-1, 1])
    batch_size = tf.shape(embeddings)[0]

    logits = tf.matmul(embeddings, embeddings, transpose_b=True) / temperature
    logits = logits - tf.reduce_max(logits, axis=1, keepdims=True)

    same_label = tf.equal(labels, tf.transpose(labels))
    self_mask = tf.eye(batch_size, dtype=tf.bool)

    positive_mask = tf.logical_and(same_label, tf.logical_not(self_mask))
    valid_mask = tf.logical_not(self_mask)

    exp_logits = tf.exp(logits) * tf.cast(valid_mask, tf.float32)
    log_prob = logits - tf.math.log(tf.reduce_sum(exp_logits, axis=1, keepdims=True) + 1e-8)

    positive_mask_f = tf.cast(positive_mask, tf.float32)
    positive_count = tf.reduce_sum(positive_mask_f, axis=1)

    mean_log_prob_pos = tf.reduce_sum(positive_mask_f * log_prob, axis=1) / tf.maximum(positive_count, 1.0)

    valid_rows = positive_count > 0.0
    loss_vec = -mean_log_prob_pos
    loss_vec = tf.boolean_mask(loss_vec, valid_rows)

    return tf.cond(tf.size(loss_vec) > 0, lambda: tf.reduce_mean(loss_vec), lambda: tf.constant(0.0, dtype=tf.float32),)


def task_alignment_loss(student_task_embedding, teacher_task_embedding):
    z_s = tf.math.l2_normalize(tf.cast(student_task_embedding, tf.float32), axis=-1)
    z_t = tf.math.l2_normalize(tf.cast(teacher_task_embedding, tf.float32), axis=-1)
    cos_sim = tf.reduce_sum(z_s * z_t, axis=-1)
    return tf.reduce_mean(1.0 - cos_sim)


def masked_mean(values, mask):
    values = tf.cast(values, tf.float32)
    mask = tf.cast(mask, tf.float32)
    return tf.reduce_sum(values * mask) / (tf.reduce_sum(mask) + 1e-8)


def task_aware_router_penalties(gate_probs, task_id, new_task_id, num_old_experts):
    """
    Penalidades para preservar tarefas antigas durante Stage 2.

    - old_task_new_expert_penalty:
        penaliza quando amostras antigas usam experts novos.

    - new_task_old_expert_penalty:
        penaliza quando a nova tarefa usa experts antigos.

    gate_probs: [B, K]
    task_id: [B]
    new_task_id: id inteiro da tarefa incremental, ex. pedestrian
    num_old_experts: número de experts do Stage 1
    """
    gate_probs = tf.cast(gate_probs, tf.float32)
    task_id = tf.cast(task_id, tf.int32)
    new_task_id = tf.cast(new_task_id, tf.int32)

    old_task_mask = tf.not_equal(task_id, new_task_id)
    new_task_mask = tf.equal(task_id, new_task_id)

    # probabilidade total enviada aos experts antigos e novos
    old_expert_prob = tf.reduce_sum(gate_probs[:, :num_old_experts], axis=-1)
    new_expert_prob = tf.reduce_sum(gate_probs[:, num_old_experts:], axis=-1)

    old_task_new_expert_penalty = masked_mean(new_expert_prob, old_task_mask)
    new_task_old_expert_penalty = masked_mean(old_expert_prob, new_task_mask)

    return old_task_new_expert_penalty, new_task_old_expert_penalty


def infer_task_id_from_mapping(task_mapping, task_name):
    """
    Tenta descobrir o id da tarefa incremental a partir de dataset.task_to_id.
    Aceita match exato, basename e substring.
    """
    if task_name is None:
        return None

    target = str(task_name).lower()

    for k, v in task_mapping.items():
        if str(k).lower() == target:
            return int(v)

    for k, v in task_mapping.items():
        base = os.path.basename(str(k)).lower()
        if base == target:
            return int(v)

    for k, v in task_mapping.items():
        base = os.path.basename(str(k)).lower()
        full = str(k).lower()
        if target in base or target in full:
            return int(v)

    return None


def get_decay_weight(epoch, initial_weight, final_weight, decay_epochs):
    """
    Decai linearmente um peso de initial_weight para final_weight.
    Use isso no começo do Stage 2 para evitar colapso do router.
    """
    if decay_epochs <= 0:
        return float(final_weight)

    progress = min(1.0, float(epoch) / float(decay_epochs))
    return float(initial_weight) * (1.0 - progress) + float(final_weight) * progress


def main():
    parser = get_argument()

    parser.add_argument("--dataset-dirs", nargs="+", required=True)
    parser.add_argument("--replay-dirs", nargs="*", default=None)
    parser.add_argument("--init-ckpt", type=str, required=True)

    parser.add_argument("--phase", type=int, default=1, choices=[1, 2])

    parser.add_argument("--student-logdir", type=str, default="checkpoints")
    parser.add_argument("--student-name", type=str, default="stage2_moe_ped_phase1")
    parser.add_argument("--student-epochs", type=int, default=100)
    parser.add_argument("--student-batch-size", type=int, default=64)
    parser.add_argument("--student-lr", type=float, default=1e-4)
    parser.add_argument("--student-target-mode", choices=["mean_action", "raw_mean"], default="mean_action")
    parser.add_argument("--student-use-vision", action="store_true")
    parser.add_argument("--student-vision-dim", type=int, default=280)
    parser.add_argument("--student-fusion-type", choices=["cross", "self"], default="cross")
    parser.add_argument("--student-save-every", type=int, default=5)

    parser.add_argument("--num-old-experts", type=int, default=2)
    parser.add_argument("--num-new-experts", type=int, default=1)
    parser.add_argument("--task-dim", type=int, default=16)
    parser.add_argument("--geo-dim", type=int, default=8)
    parser.add_argument("--geo-type", type=str, default="mlp", choices=["mlp", "cross_attn"])


    parser.add_argument("--router-balance-weight", type=float, default=0.01)
    parser.add_argument("--router-entropy-weight", type=float, default=0.0)

    parser.add_argument("--steer-weight", type=float, default=2.0)
    parser.add_argument("--speed-weight", type=float, default=1.0)

    parser.add_argument("--use-mode-contrastive", action="store_true")
    parser.add_argument("--mode-contrastive-weight", type=float, default=0.0)
    parser.add_argument("--mode-contrastive-temp", type=float, default=0.1)
    parser.add_argument("--use-task-contrastive", action="store_true")
    parser.add_argument("--task-contrastive-weight", type=float, default=0.0)
    parser.add_argument("--task-contrastive-temp", type=float, default=0.1)
    parser.add_argument("--int-dim", type=int, default=8)

    parser.add_argument("--use-int-contrastive", action="store_true")
    parser.add_argument("--int-contrastive-weight", type=float, default=0.0)
    parser.add_argument("--int-contrastive-temp", type=float, default=0.1)

    parser.add_argument("--interaction-type", type=str, default="mlp", choices=["mlp", "cross_attn"],)

    parser.add_argument("--use-geo", action="store_true")
    parser.add_argument("--use-int", action="store_true")

    parser.add_argument("--init-new-expert-from-old-avg", action="store_true")
    parser.add_argument("--new-expert-noise-std", type=float, default=1e-3)

    parser.add_argument("--teacher-task-prototypes", type=str, default=None)
    parser.add_argument("--use-task-alignment", action="store_true")
    parser.add_argument("--task-alignment-weight", type=float, default=0.0)

    # Proteção do roteamento para evitar que o expert novo domine tarefas antigas.
    # Estratégia 2: penaliza uso de expert novo em tarefas antigas.
    # Estratégia 3: também incentiva a nova tarefa a usar o expert novo.
    parser.add_argument("--new-task-name", type=str, default="pedestrian")
    parser.add_argument("--router-new-task-id", type=int, default=-1)
    parser.add_argument("--old-task-new-expert-penalty-weight", type=float, default=0.0)
    parser.add_argument("--new-task-old-expert-penalty-weight", type=float, default=0.0)

    # Warm-up/decay do router. Se estes argumentos não forem passados,
    # o comportamento antigo é preservado usando router_balance_weight e router_entropy_weight.
    parser.add_argument("--router-balance-initial-weight", type=float, default=None)
    parser.add_argument("--router-balance-final-weight", type=float, default=None)
    parser.add_argument("--router-balance-warmup-epochs", type=int, default=0)

    parser.add_argument("--router-entropy-initial-weight", type=float, default=None)
    parser.add_argument("--router-entropy-final-weight", type=float, default=None)
    parser.add_argument("--router-entropy-warmup-epochs", type=int, default=0)

    args = parser.parse_args()
    args, algo_params, _ = set_configs(args, test=False)

    # Mantém compatibilidade com comandos antigos:
    # se o warm-up não for configurado, usa pesos fixos como antes.
    if args.router_balance_initial_weight is None:
        args.router_balance_initial_weight = args.router_balance_weight
    if args.router_balance_final_weight is None:
        args.router_balance_final_weight = args.router_balance_weight
    if args.router_entropy_initial_weight is None:
        args.router_entropy_initial_weight = args.router_entropy_weight
    if args.router_entropy_final_weight is None:
        args.router_entropy_final_weight = args.router_entropy_weight

    gpus = tf.config.experimental.list_physical_devices("GPU")
    if args.gpu is not None and args.gpu >= 0 and len(gpus) > args.gpu:
        tf.config.set_visible_devices([gpus[args.gpu]], "GPU")
        tf.config.experimental.set_memory_growth(gpus[args.gpu], True)
        print("Usando GPU:", gpus[args.gpu])
    else:
        tf.config.set_visible_devices([], "GPU")
        print("Treinando na CPU.")

    replay_dirs = args.replay_dirs if args.replay_dirs is not None else []
    all_dataset_dirs = list(args.dataset_dirs) + list(replay_dirs)

    dataset = DistillTransitionDataset(dataset_dirs=all_dataset_dirs,
                                        target_mode=args.student_target_mode,
                                        use_vision=args.student_use_vision,
                                        require_map=True,
                                        teacher_task_prototypes_path=args.teacher_task_prototypes,
                                      )

    if args.router_new_task_id >= 0:
        resolved_new_task_id = int(args.router_new_task_id)
    else:
        resolved_new_task_id = infer_task_id_from_mapping(dataset.task_to_id, args.new_task_name)
        if resolved_new_task_id is None:
            raise ValueError(
                f"Could not infer new_task_id for new_task_name={args.new_task_name}. "
                f"Available task_mapping={dataset.task_to_id}. "
                f"Please pass --router-new-task-id explicitly."
            )

    print(f"[Stage2MoE] task_mapping: {dataset.task_to_id}")
    print(f"[Stage2MoE] new task for router protection: {args.new_task_name} -> id={resolved_new_task_id}")

    train_ds = dataset.make_tf_dataset(batch_size=args.student_batch_size,
                                       shuffle=True,
                                       shuffle_buffer=10000,
                                       repeat=False,
                                       )

    sample0 = dataset.get_specimen()
    state_shape = sample0["obs"].shape
    action_dim = sample0["act"].shape[-1]

    student_params = build_student_params(args, algo_params)

    policy = StudentMoEStage2(params=student_params,
                              state_shape=state_shape,
                              action_dim=action_dim,
                              max_action=1.0,
                              num_old_experts=args.num_old_experts,
                              num_new_experts=args.num_new_experts,
                              task_dim=args.task_dim,
                              geo_dim=args.geo_dim,
                              int_dim=args.int_dim,
                              use_geo=args.use_geo,
                              use_int=args.use_int,
                              geo_type=args.geo_type,
                              interaction_type=args.interaction_type,
                            )
    policy.summary()

    # stage1_model = build_stage1_temp_model(params=student_params,
    #                                        state_shape=state_shape,
    #                                        action_dim=action_dim,
    #                                        num_old_experts=args.num_old_experts,
    #                                        task_dim=args.task_dim,
    #                                       )
    
    # print("===== DEBUG BEFORE RESTORE =====")
    # print("args.init_ckpt:", args.init_ckpt)
    # print("args.num_old_experts:", args.num_old_experts)
    # print("args.num_new_experts:", args.num_new_experts)

    # ckpt_reader = tf.train.load_checkpoint(args.init_ckpt)
    # print("checkpoint router kernel shape:",
    #     ckpt_reader.get_tensor("student/router/logits/kernel/.ATTRIBUTES/VARIABLE_VALUE").shape)
    # print("checkpoint router bias shape:",
    #     ckpt_reader.get_tensor("student/router/logits/bias/.ATTRIBUTES/VARIABLE_VALUE").shape)

    # print("len(stage1_model.moe.experts):", len(stage1_model.moe.experts))
    # print("stage1_model router kernel shape:", stage1_model.router.logits.kernel.shape)
    # print("stage1_model router bias shape:", stage1_model.router.logits.bias.shape)
    # print("===== END DEBUG =====")

    # restored_ckpt = restore_stage1_moe(stage1_model, args.init_ckpt)
    # copy_weights_stage1_to_stage2(stage1_model, policy)

    # policy.configure_phase(args.phase)

    # if args.phase == 1:
    #     stage1_model = build_stage1_temp_model(params=student_params,
    #                                            state_shape=state_shape,
    #                                            action_dim=action_dim,
    #                                            num_old_experts=args.num_old_experts,
    #                                            task_dim=args.task_dim,
    #                                           )

    #     restored_ckpt = restore_stage1_moe(stage1_model, args.init_ckpt)
    #     copy_weights_stage1_to_stage2(stage1_model, policy)


    if args.phase == 1:
        # modelo temporário do Stage 1 antigo
        stage1_model = build_stage1_temp_model(
            params=student_params,
            state_shape=state_shape,
            action_dim=action_dim,
            num_old_experts=args.num_old_experts,
            task_dim=args.task_dim,
        )

        restored_ckpt = restore_stage1_moe(stage1_model, args.init_ckpt)

        # copia parcial para a nova arquitetura zint
        copy_weights_stage1_to_stage2(stage1_model, policy)
        if args.init_new_expert_from_old_avg:
            initialize_new_experts_from_old_average(stage2_model=policy,
                                                    noise_std=args.new_expert_noise_std,
                                                   )

        print(f"[Stage2MoE] Phase 1 initialized from Stage1 checkpoint: {restored_ckpt}")


    elif args.phase == 2:
        restore_ckpt = args.init_ckpt
        if tf.io.gfile.isdir(restore_ckpt):
            latest = tf.train.latest_checkpoint(restore_ckpt)
            if latest is None:
                raise ValueError(f"No checkpoint found inside directory: {restore_ckpt}")
            restore_ckpt = latest

        ckpt_restore = tf.train.Checkpoint(student=policy)
        ckpt_restore.restore(restore_ckpt).expect_partial()
        restored_ckpt = restore_ckpt
        # print(f"[Stage2MoE] Restored stage2 checkpoint: {restored_ckpt}")
        print(f"[Stage2MoE-ZINT] Restored stage2 checkpoint: {restored_ckpt}")

    policy.configure_phase(args.phase)


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
        json.dump({"phase": args.phase,
                   "dataset_dirs": args.dataset_dirs,
                   "replay_dirs": replay_dirs,
                   "all_dataset_dirs": all_dataset_dirs,
                   "student_target_mode": args.student_target_mode,
                   "student_use_vision": args.student_use_vision,
                   "student_vision_dim": args.student_vision_dim,
                   "student_fusion_type": args.student_fusion_type,
                   "num_old_experts": args.num_old_experts,
                   "num_new_experts": args.num_new_experts,
                   "task_dim": args.task_dim,
                   "geo_dim": args.geo_dim,
                   "geo_type": args.geo_type,
                   "use_geo": args.use_geo,
                   "use_int": args.use_int,
                   "router_balance_weight": args.router_balance_weight,
                   "router_entropy_weight": args.router_entropy_weight,
                   "init_ckpt": args.init_ckpt,
                   "restored_ckpt": restored_ckpt,
                   "start_time": start_dt.isoformat(),
                   "task_mapping": dataset.task_to_id,
                   "speed_weight": args.speed_weight,
                   "steer_weight": args.steer_weight,
                   "use_mode_contrastive": args.use_mode_contrastive,
                   "mode_contrastive_weight": args.mode_contrastive_weight,
                   "mode_contrastive_temp": args.mode_contrastive_temp,
                   "use_task_contrastive": args.use_task_contrastive,
                   "task_contrastive_weight": args.task_contrastive_weight,
                   "task_contrastive_temp": args.task_contrastive_temp,
                   "int_dim": args.int_dim,
                   "use_int_contrastive": args.use_int_contrastive,
                   "int_contrastive_weight": args.int_contrastive_weight,
                   "int_contrastive_temp": args.int_contrastive_temp,
                   "interaction_type": args.interaction_type,
                   "init_new_expert_from_old_avg": args.init_new_expert_from_old_avg,
                   "new_expert_noise_std": args.new_expert_noise_std,
                   "teacher_task_prototypes": args.teacher_task_prototypes,
                   "use_task_alignment": args.use_task_alignment,
                   "task_alignment_weight": args.task_alignment_weight,
                   "new_task_name": args.new_task_name,
                   "router_new_task_id": int(resolved_new_task_id),
                   "old_task_new_expert_penalty_weight": args.old_task_new_expert_penalty_weight,
                   "new_task_old_expert_penalty_weight": args.new_task_old_expert_penalty_weight,
                   "router_balance_initial_weight": args.router_balance_initial_weight,
                   "router_balance_final_weight": args.router_balance_final_weight,
                   "router_balance_warmup_epochs": args.router_balance_warmup_epochs,
                   "router_entropy_initial_weight": args.router_entropy_initial_weight,
                   "router_entropy_final_weight": args.router_entropy_final_weight,
                   "router_entropy_warmup_epochs": args.router_entropy_warmup_epochs,
                   }, 
                   f, 
                   indent=2
                  )
        

    
    
    @tf.function
    def train_step(batch, current_router_balance_weight, current_router_entropy_weight):
        obs = tf.convert_to_tensor(batch["obs"], dtype=tf.float32)
        mask = tf.convert_to_tensor(batch["mask"], dtype=tf.float32)
        map_state = batch_value_or_none(batch, "map_state")
        vision = batch_value_or_none(batch, "vision")
        target = tf.convert_to_tensor(batch["target"], dtype=tf.float32)

        # por enquanto o dataset ainda chama isso de mode_label,
        # mas semanticamente agora ele funciona como geo_label
        geo_label = tf.convert_to_tensor(batch["mode_label"], dtype=tf.int32)
        task_id = tf.convert_to_tensor(batch["task_id"], dtype=tf.int32)
        interaction_label = tf.convert_to_tensor(batch["interaction_label"], dtype=tf.int32)
        teacher_task_embedding = batch_value_or_none(batch, "teacher_task_embedding")

        with tf.GradientTape() as tape:
            out = policy(
                obs,
                mask=mask,
                map_state=map_state,
                vision=vision,
                training=True,
                return_aux=True,
            )

            pred = out["raw_mean"] if args.student_target_mode == "raw_mean" else out["action"]

            distill_loss, mae, speed_mse, steer_mse = weighted_action_loss(pred,
                                                                           target,
                                                                           speed_weight=args.speed_weight,
                                                                           steer_weight=args.steer_weight,
                                                                           )

            balance_loss = router_balance_loss(out["gate_probs"])
            ent = router_entropy(out["gate_probs"])

            total_loss = distill_loss
            total_loss += tf.cast(current_router_balance_weight, tf.float32) * balance_loss
            total_loss += tf.cast(current_router_entropy_weight, tf.float32) * ent

            old_task_new_expert_penalty = tf.constant(0.0, dtype=tf.float32)
            new_task_old_expert_penalty = tf.constant(0.0, dtype=tf.float32)

            if (args.old_task_new_expert_penalty_weight > 0.0
                or args.new_task_old_expert_penalty_weight > 0.0):
                (old_task_new_expert_penalty,
                 new_task_old_expert_penalty) = task_aware_router_penalties(
                    gate_probs=out["gate_probs"],
                    task_id=task_id,
                    new_task_id=tf.constant(resolved_new_task_id, dtype=tf.int32),
                    num_old_experts=args.num_old_experts,
                )

                total_loss += args.old_task_new_expert_penalty_weight * old_task_new_expert_penalty
                total_loss += args.new_task_old_expert_penalty_weight * new_task_old_expert_penalty

            geo_ctr_loss = tf.constant(0.0, dtype=tf.float32)
            if args.use_geo and args.use_mode_contrastive and args.mode_contrastive_weight > 0.0:
                geo_ctr_loss = supervised_contrastive_loss(
                    out["geo_embedding"],
                    geo_label,
                    temperature=args.mode_contrastive_temp,
                )
                total_loss += args.mode_contrastive_weight * geo_ctr_loss

            task_ctr_loss = tf.constant(0.0, dtype=tf.float32)
            task_align_loss = tf.constant(0.0, dtype=tf.float32)

            if args.use_task_contrastive and args.task_contrastive_weight > 0.0:
                task_ctr_loss = supervised_contrastive_loss(
                    out["task_embedding"],
                    task_id,
                    temperature=args.task_contrastive_temp,
                )
                total_loss += args.task_contrastive_weight * task_ctr_loss

            if (args.use_task_alignment and args.task_alignment_weight > 0.0 and teacher_task_embedding is not None):
                task_align_loss = task_alignment_loss(out["task_embedding"],teacher_task_embedding,)
                total_loss += args.task_alignment_weight * task_align_loss

            int_ctr_loss = tf.constant(0.0, dtype=tf.float32)
            if args.use_int and args.use_int_contrastive and args.int_contrastive_weight > 0.0:
                int_ctr_loss = supervised_contrastive_loss(out["interaction_embedding"],
                                                           interaction_label,
                                                           temperature=args.int_contrastive_temp,
                                                           )
                total_loss += args.int_contrastive_weight * int_ctr_loss

        grads = tape.gradient(total_loss, policy.trainable_variables)
        optimizer.apply_gradients(zip(grads, policy.trainable_variables))

        return (total_loss,
               distill_loss,
               mae,
               balance_loss,
               ent,
               old_task_new_expert_penalty,
               new_task_old_expert_penalty,
               geo_ctr_loss,
               task_ctr_loss,
               task_align_loss,
               int_ctr_loss,
               speed_mse,
               steer_mse,
               out["task_embedding"],
               out["geo_embedding"],
               out["interaction_embedding"],
              out["gate_probs"],
            )
                


    # @tf.function
    # def train_step(batch):
    #     obs = tf.convert_to_tensor(batch["obs"], dtype=tf.float32)
    #     mask = tf.convert_to_tensor(batch["mask"], dtype=tf.float32)
    #     map_state = batch_value_or_none(batch, "map_state")
    #     vision = batch_value_or_none(batch, "vision")
    #     target = tf.convert_to_tensor(batch["target"], dtype=tf.float32)
    #     mode_label = tf.convert_to_tensor(batch["mode_label"], dtype=tf.int32)
    #     task_id = tf.convert_to_tensor(batch["task_id"], dtype=tf.int32)
    #     interaction_label = tf.convert_to_tensor(batch["interaction_label"], dtype=tf.int32)

    #     with tf.GradientTape() as tape:
    #         out = policy(obs, mask=mask, map_state=map_state, vision=vision, training=True, return_aux=True,)

    #         # pred = out["raw_mean"] if args.student_target_mode == "raw_mean" else out["action"]

    #         # distill_loss = tf.reduce_mean(tf.square(pred - target))
    #         # mae = tf.reduce_mean(tf.abs(pred - target))

    #         # balance_loss = router_balance_loss(out["gate_probs"])
    #         # ent = router_entropy(out["gate_probs"])

    #         # total_loss = distill_loss
    #         # total_loss += args.router_balance_weight * balance_loss
    #         # total_loss += args.router_entropy_weight * ent

    #         pred = out["raw_mean"] if args.student_target_mode == "raw_mean" else out["action"]

    #         distill_loss, mae, speed_mse, steer_mse = weighted_action_loss(pred,
    #                                                                        target,
    #                                                                        speed_weight=args.speed_weight,
    #                                                                        steer_weight=args.steer_weight,
    #                                                                        )

    #         balance_loss = router_balance_loss(out["gate_probs"])
    #         ent = router_entropy(out["gate_probs"])

    #         total_loss = distill_loss
    #         total_loss += args.router_balance_weight * balance_loss
    #         total_loss += args.router_entropy_weight * ent

    #         mode_ctr_loss = tf.constant(0.0, dtype=tf.float32)
    #         if args.use_mode_contrastive and args.mode_contrastive_weight > 0.0:
    #             mode_ctr_loss = supervised_contrastive_loss(out["mode_embedding"],
    #                                                         mode_label,
    #                                                         temperature=args.mode_contrastive_temp,
    #                                                         )
                
    #             total_loss += args.mode_contrastive_weight * mode_ctr_loss

            
    #         task_ctr_loss = tf.constant(0.0, dtype=tf.float32)
    #         if args.use_task_contrastive and args.task_contrastive_weight > 0.0:
    #             task_ctr_loss = supervised_contrastive_loss(out["task_embedding"],
    #                                                         task_id,
    #                                                         temperature=args.task_contrastive_temp,
    #                                                         )
                
    #             total_loss += args.task_contrastive_weight * task_ctr_loss

            
    #         int_ctr_loss = tf.constant(0.0, dtype=tf.float32)
    #         if args.use_int_contrastive and args.int_contrastive_weight > 0.0:
    #             int_ctr_loss = supervised_contrastive_loss(out["interaction_embedding"],
    #                                                        interaction_label,
    #                                                        temperature=args.int_contrastive_temp,
    #                                                        )
                
    #             total_loss += args.int_contrastive_weight * int_ctr_loss


    #     grads = tape.gradient(total_loss, policy.trainable_variables)
    #     optimizer.apply_gradients(zip(grads, policy.trainable_variables))
    #     # return total_loss, distill_loss, mae, balance_loss, ent
    #     return (total_loss, 
    #             distill_loss, 
    #             mae, 
    #             balance_loss, 
    #             ent, 
    #             mode_ctr_loss,
    #             task_ctr_loss,
    #             int_ctr_loss,
    #             speed_mse, 
    #             steer_mse,
    #             out["task_embedding"],
    #             out["mode_embedding"],
    #             out["interaction_embedding"],
    #             out["gate_probs"],
    #             )

    for epoch in range(1, args.student_epochs + 1):
        epoch_start = time.perf_counter()

        current_router_balance_weight = get_decay_weight(
            epoch=epoch,
            initial_weight=args.router_balance_initial_weight,
            final_weight=args.router_balance_final_weight,
            decay_epochs=args.router_balance_warmup_epochs,
        )

        current_router_entropy_weight = get_decay_weight(
            epoch=epoch,
            initial_weight=args.router_entropy_initial_weight,
            final_weight=args.router_entropy_final_weight,
            decay_epochs=args.router_entropy_warmup_epochs,
        )

        int_ctr_losses = []
        # total_losses = []
        # distill_losses = []
        # maes = []
        # balances = []
        # ents = []

        # for batch in train_ds:
        #     total_loss, distill_loss, mae, balance_loss, ent = train_step(batch)
        #     total_losses.append(float(total_loss.numpy()))
        #     distill_losses.append(float(distill_loss.numpy()))
        #     maes.append(float(mae.numpy()))
        #     balances.append(float(balance_loss.numpy()))
        #     ents.append(float(ent.numpy()))

        total_losses = []
        distill_losses = []
        maes = []
        balances = []
        ents = []
        old_task_new_expert_penalties = []
        new_task_old_expert_penalties = []
        speed_mses = []
        steer_mses = []
        geo_ctr_losses = []
        task_ctr_losses = []
        task_align_losses = []

        task_embeds_epoch = []
        geo_embeds_epoch = []
        gate_probs_epoch = []

        task_ids_epoch = []
        geo_labels_epoch = []

        interaction_embeds_epoch = []
        interaction_labels_epoch = []

        for batch in train_ds:

            (total_loss, 
            distill_loss, 
            mae, 
            balance_loss, 
            ent, 
            old_task_new_expert_penalty,
            new_task_old_expert_penalty,
            geo_ctr_loss,
            task_ctr_loss,
            task_align_loss,
            int_ctr_loss,
            speed_mse, 
            steer_mse, 
            task_embed,
            geo_embed,
            interaction_embed,
            gate_probs,
            ) = train_step(
                batch,
                tf.constant(current_router_balance_weight, dtype=tf.float32),
                tf.constant(current_router_entropy_weight, dtype=tf.float32),
            )
            
            total_losses.append(float(total_loss.numpy()))
            distill_losses.append(float(distill_loss.numpy()))
            maes.append(float(mae.numpy()))
            balances.append(float(balance_loss.numpy()))
            ents.append(float(ent.numpy()))
            old_task_new_expert_penalties.append(float(old_task_new_expert_penalty.numpy()))
            new_task_old_expert_penalties.append(float(new_task_old_expert_penalty.numpy()))
            speed_mses.append(float(speed_mse.numpy()))
            steer_mses.append(float(steer_mse.numpy()))
            geo_ctr_losses.append(float(geo_ctr_loss.numpy()))
            task_ctr_losses.append(float(task_ctr_loss.numpy()))
            task_align_losses.append(float(task_align_loss.numpy()))

            task_embeds_epoch.append(task_embed.numpy())
            gate_probs_epoch.append(gate_probs.numpy())
            task_ids_epoch.append(np.array(batch["task_id"]))

            if geo_embed is not None:
                geo_embeds_epoch.append(geo_embed.numpy())
                geo_labels_epoch.append(np.array(batch["mode_label"]))

            if interaction_embed is not None:
                interaction_embeds_epoch.append(interaction_embed.numpy())
                interaction_labels_epoch.append(np.array(batch["interaction_label"]))

            int_ctr_losses.append(float(int_ctr_loss.numpy()))

        row = {"epoch": epoch,
               "total_loss": float(np.mean(total_losses)),
               "distill_mse": float(np.mean(distill_losses)),
               "action_mae": float(np.mean(maes)),
               "router_balance_loss": float(np.mean(balances)),
               "router_entropy": float(np.mean(ents)),
               "old_task_new_expert_penalty": float(np.mean(old_task_new_expert_penalties)),
               "new_task_old_expert_penalty": float(np.mean(new_task_old_expert_penalties)),
               "epoch_duration_sec": float(time.perf_counter() - epoch_start),
               "speed_mse": float(np.mean(speed_mses)),
               "steer_mse": float(np.mean(steer_mses)),
               "geo_contrastive_loss": float(np.mean(geo_ctr_losses)),
               "task_contrastive_loss": float(np.mean(task_ctr_losses)),
               "task_alignment_loss": float(np.mean(task_align_losses)),
               "int_contrastive_loss": float(np.mean(int_ctr_losses)),
               "router_balance_weight": float(current_router_balance_weight),
               "router_entropy_weight": float(current_router_entropy_weight),
               }
        metrics.append(row)

        with tb_train.as_default():
            tf.summary.scalar("train/total_loss", row["total_loss"], step=epoch)
            tf.summary.scalar("train/distill_mse", row["distill_mse"], step=epoch)
            tf.summary.scalar("train/action_mae", row["action_mae"], step=epoch)
            tf.summary.scalar("train/router_balance_loss", row["router_balance_loss"], step=epoch)
            tf.summary.scalar("train/router_entropy", row["router_entropy"], step=epoch)
            tf.summary.scalar("train/old_task_new_expert_penalty", row["old_task_new_expert_penalty"], step=epoch)
            tf.summary.scalar("train/new_task_old_expert_penalty", row["new_task_old_expert_penalty"], step=epoch)
            tf.summary.scalar("train/epoch_duration_sec", row["epoch_duration_sec"], step=epoch)
            tf.summary.scalar("train/speed_mse", row["speed_mse"], step=epoch)
            tf.summary.scalar("train/steer_mse", row["steer_mse"], step=epoch)
            tf.summary.scalar("train/geo_contrastive_loss", row["geo_contrastive_loss"], step=epoch)
            tf.summary.scalar("train/task_contrastive_loss", row["task_contrastive_loss"], step=epoch)
            tf.summary.scalar("train/task_alignment_loss", row["task_alignment_loss"], step=epoch)
            tf.summary.scalar("train/int_contrastive_loss", row["int_contrastive_loss"], step=epoch)
            tf.summary.scalar("train/router_balance_weight", row["router_balance_weight"], step=epoch)
            tf.summary.scalar("train/router_entropy_weight", row["router_entropy_weight"], step=epoch)
        tb_train.flush()

        print(f"[Stage2MoE][Phase {args.phase}] epoch={epoch:03d} | "
              f"total_loss={row['total_loss']:.6f} | "
              f"distill_mse={row['distill_mse']:.6f} | "
              f"geo_ctr={row['geo_contrastive_loss']:.6f} | "
              f"task_ctr={row['task_contrastive_loss']:.6f} | "
              f"task_align={row['task_alignment_loss']:.6f} | "
              f"speed_mse={row['speed_mse']:.6f} | "
              f"steer_mse={row['steer_mse']:.6f} | "
              # f"action_mae={row['action_mae']:.6f} | "
              f"router_balance={row['router_balance_loss']:.6f} | "
              f"router_balance_w={row['router_balance_weight']:.6f} | "
              f"old2new_pen={row['old_task_new_expert_penalty']:.6f} | "
              f"new2old_pen={row['new_task_old_expert_penalty']:.6f} | "
              # f"router_entropy={row['router_entropy']:.6f} | "
              # f"epoch_duration={row['epoch_duration_sec']:.1f} sec | "
              f"int_ctr={row['int_contrastive_loss']:.6f} | "
              )

        save_rows_json(os.path.join(ckpt_dir, "train_metrics.json"), metrics)
        save_rows_csv(os.path.join(ckpt_dir, "train_metrics.csv"), metrics)

        if epoch % args.student_save_every == 0 or epoch == args.student_epochs:
            path = manager.save()
            print(f"Saved Stage2MoE checkpoint: {path}")

            embed_dir = os.path.join(ckpt_dir, "embeddings")
            os.makedirs(embed_dir, exist_ok=True)

            task_np = np.concatenate(task_embeds_epoch, axis=0)
            gate_np = np.concatenate(gate_probs_epoch, axis=0)
            expert_id = np.argmax(gate_np, axis=-1)
            task_ids_np = np.concatenate(task_ids_epoch, axis=0)

            save_dict = {
                "task_embedding": task_np,
                "gate_probs": gate_np,
                "expert_id": expert_id,
                "task_id": task_ids_np,
            }

            if len(geo_embeds_epoch) > 0:
                save_dict["geo_embedding"] = np.concatenate(geo_embeds_epoch, axis=0)
                save_dict["geo_label"] = np.concatenate(geo_labels_epoch, axis=0)

            if len(interaction_embeds_epoch) > 0:
                save_dict["interaction_embedding"] = np.concatenate(interaction_embeds_epoch, axis=0)
                save_dict["interaction_label"] = np.concatenate(interaction_labels_epoch, axis=0)

            np.savez_compressed(
                os.path.join(embed_dir, f"epoch_{epoch:03d}_embeddings.npz"),
                **save_dict
            )

    end_dt = datetime.now()
    with open(os.path.join(ckpt_dir, "run_summary.json"), "w") as f:
        json.dump({"phase": args.phase,
                   "train_start_time": start_dt.isoformat(),
                   "train_end_time": end_dt.isoformat(),
                   "total_train_duration_sec": float(time.perf_counter() - start_perf),
                   "best_distill_mse": float(np.min([m["distill_mse"] for m in metrics])) if metrics else None,
                   "last_distill_mse": metrics[-1]["distill_mse"] if metrics else None,
                   "last_action_mae": metrics[-1]["action_mae"] if metrics else None,
                   "speed_weight": args.speed_weight,
                   "steer_weight": args.steer_weight,
                   "use_mode_contrastive": args.use_mode_contrastive,
                   "mode_contrastive_weight": args.mode_contrastive_weight,
                   "mode_contrastive_temp": args.mode_contrastive_temp,
                   "int_dim": args.int_dim,
                   "use_int_contrastive": args.use_int_contrastive,
                   "int_contrastive_weight": args.int_contrastive_weight,
                   "int_contrastive_temp": args.int_contrastive_temp,
                   "init_new_expert_from_old_avg": args.init_new_expert_from_old_avg,
                   "new_expert_noise_std": args.new_expert_noise_std,
                   "teacher_task_prototypes": args.teacher_task_prototypes,
                   "use_task_alignment": args.use_task_alignment,
                   "task_alignment_weight": args.task_alignment_weight,
                   "new_task_name": args.new_task_name,
                   "router_new_task_id": int(resolved_new_task_id),
                   "old_task_new_expert_penalty_weight": args.old_task_new_expert_penalty_weight,
                   "new_task_old_expert_penalty_weight": args.new_task_old_expert_penalty_weight,
                   "router_balance_initial_weight": args.router_balance_initial_weight,
                   "router_balance_final_weight": args.router_balance_final_weight,
                   "router_balance_warmup_epochs": args.router_balance_warmup_epochs,
                   "router_entropy_initial_weight": args.router_entropy_initial_weight,
                   "router_entropy_final_weight": args.router_entropy_final_weight,
                   "router_entropy_warmup_epochs": args.router_entropy_warmup_epochs,
                   }, 
                   f, 
                   indent=2
                  )


if __name__ == "__main__":
    main()



# python train_student_moe_stage2.py \
#   --dataset-dirs ../../export_rollouts/datasets/pedestrian \
#   --replay-dirs ../../export_rollouts/datasets/lane_keeping_replay_10pct_success_curve \
#   --init-ckpt ../baseline_obs/checkpoints/stage1_moe_lane_steer2/ckpt-5 \
#   --phase 1 \
#   --student-logdir checkpoints_zint \
#   --student-name stage2_moe_ped_phase1_zint \
#   --student-epochs 80 \
#   --student-batch-size 64 \
#   --student-lr 1e-4 \
#   --student-target-mode mean_action \
#   --num-old-experts 2 \
#   --num-new-experts 1 \
#   --task-dim 16 \
#   --mode-dim 8 \
#   --int-dim 8 \
#   --interaction-type cross_attn \
#   --router-balance-weight 0.01 \
#   --speed-weight 1.0 \
#   --steer-weight 2.0



# python train_student_moe_stage2.py \
#   --dataset-dirs ../../export_rollouts/datasets/pedestrian \
#   --replay-dirs ../../export_rollouts/datasets/lane_keeping_replay_10pct_success_curve \
#   --init-ckpt checkpoints_zint/stage2_moe_ped_phase1_zint/ckpt-10 \
#   --phase 2 \
#   --student-logdir checkpoints_zint \
#   --student-name stage2_moe_ped_phase2_zint \
#   --student-epochs 80 \
#   --student-batch-size 64 \
#   --student-lr 5e-5 \
#   --student-target-mode mean_action \
#   --num-old-experts 3 \
#   --num-new-experts 0 \
#   --task-dim 16 \
#   --mode-dim 8 \
#   --int-dim 8 \
#   --router-balance-weight 0.02 \
#   --router-entropy-weight 0.001 \
#   --speed-weight 1.0 \
#   --steer-weight 2.0 \
#   --use-mode-contrastive \
#   --mode-contrastive-weight 0.01 \
#   --mode-contrastive-temp 0.1 \
#   --use-task-contrastive \
#   --task-contrastive-weight 0.01 \
#   --task-contrastive-temp 0.1 \
#   --use-int-contrastive \
#   --int-contrastive-weight 0.01 \
#   --int-contrastive-temp 0.1
###################################################################################################
# Phase1
# python train_student_moe_stage2.py \
#   --dataset-dirs ../../export_rollouts/datasets/pedestrian \
#   --replay-dirs ../../export_rollouts/datasets/lane_keeping_replay_10pct_success_curve \
#   --init-ckpt ../baseline_obs/checkpoints/stage1_moe_lane_steer2/ckpt-5 \
#   --phase 1 \
#   --student-logdir checkpoints_zgeo_zint \
#   --student-name stage2_moe_ped_phase1_geo_int \
#   --student-epochs 80 \
#   --student-batch-size 64 \
#   --student-lr 1e-4 \
#   --student-target-mode mean_action \
#   --num-old-experts 2 \
#   --num-new-experts 1 \
#   --task-dim 16 \
#   --geo-dim 8 \
#   --int-dim 8 \
#   --geo-type cross_attn \
#   --interaction-type cross_attn \
#   --router-balance-weight 0.01 \
#   --speed-weight 1.0 \
#   --steer-weight 2.0 \
#   --use-task-contrastive \
#   --task-contrastive-weight 0.01 \
#   --task-contrastive-temp 0.1 \
#   --use-mode-contrastive \
#   --mode-contrastive-weight 0.01 \
#   --mode-contrastive-temp 0.1 \
#   --use-int-contrastive \
#   --int-contrastive-weight 0.005 \
#   --int-contrastive-temp 0.1

############################################################################################################
# Phase2
# python3 train_student_moe_stage2.py \
#   --dataset-dirs ../../export_rollouts/datasets/pedestrian \
#   --replay-dirs ../../export_rollouts/datasets/lane_keeping_replay_10pct_success_curve \
#   --init-ckpt checkpoints_zgeo_zint/stage2_moe_ped_phase1_geo_int/ckpt-16 \
#   --phase 2 \
#   --student-logdir checkpoints_zgeo_zint \
#   --student-name stage2_moe_ped_phase2_geo_int \
#   --student-epochs 100 \
#   --student-batch-size 64 \
#   --student-lr 5e-5 \
#   --student-target-mode mean_action \
#   --num-old-experts 3 \
#   --num-new-experts 0 \
#   --task-dim 16 \
#   --geo-dim 8 \
#   --int-dim 8 \
#   --geo-type cross_attn \
#   --interaction-type cross_attn \
#   --router-balance-weight 0.02 \
#   --router-entropy-weight 0.001 \
#   --speed-weight 1.0 \
#   --steer-weight 2.0 \
#   --use-task-contrastive \
#   --task-contrastive-weight 0.01 \
#   --task-contrastive-temp 0.1 \
#   --use-mode-contrastive \
#   --mode-contrastive-weight 0.01 \
#   --mode-contrastive-temp 0.1 \
#   --use-int-contrastive \
#   --int-contrastive-weight 0.01 \
#   --int-contrastive-temp 0.1
######################################################################################################################
#ETAPA2 PHASE1 WITH GEO AND INT
#######################################################################################################################
# python train_student_moe_stage2.py \
#   --dataset-dirs ../../export_rollouts/datasets/pedestrian_antigo \
#   --replay-dirs ../../export_rollouts/datasets/lane_keeping_replay_10pct_success_curve ../../export_rollouts/datasets/change_lane_replay_10pct_success \
#   --init-ckpt checkpoints_stage1_multitask/stage1_moe_lane_change/ckpt-20 \
#   --phase 1 \
#   --student-logdir checkpoints_stage2_multitask \
#   --student-name stage2_moe_ped_phase1_noprot_no_geoint \
#   --student-epochs 100 \
#   --student-batch-size 64 \
#   --student-lr 1e-4 \
#   --student-target-mode mean_action \
#   --num-old-experts 2 \
#   --num-new-experts 1 \
#   --task-dim 16 \
#   --router-balance-weight 0.01 \
#   --speed-weight 1.0 \
#   --steer-weight 2.0
##########################################################################################################################
# ETAPA2 PHASE1 WITHOUT GEO AND INT
#########################################################################################################################
# python train_student_moe_stage2.py \
#   --dataset-dirs ../../export_rollouts/datasets/pedestrian_antigo \
#   --replay-dirs ../../export_rollouts/datasets/lane_keeping_replay_10pct_success_curve ../../export_rollouts/datasets/change_lane_replay_10pct_success \
#   --init-ckpt Resultados/checkpoints_stage1_multitask/stage1_moe_lane_change/ckpt-20 \
#   --phase 1 \
#   --student-logdir Resultados/checkpoints_stage2_multitask \
#   --student-name stage2_moe_ped_phase1_noprot_no_geoint \
#   --student-epochs 100 \
#   --student-batch-size 64 \
#   --student-lr 1e-4 \
#   --student-target-mode mean_action \
#   --num-old-experts 2 \
#   --num-new-experts 1 \
#   --task-dim 16 \
#   --router-balance-weight 0.01 \
#   --speed-weight 1.0 \
#   --steer-weight 2.0