from email import parser
import os
import sys
sys.path.append('../../../')

import csv
import json

import numpy as np
import tensorflow as tf
import gym

from envs.carla_route_env import CarlaRouteEnv, EnvConfig
from train.init_configs import get_argument, set_configs
from tasks.lane_keeping import LaneKeepingTask
from tasks.pedestrian import PedestrianTask
from tasks.change_lane import ChangeLaneTask

from student_model_moe import StudentMoE
from student_model_moe_stage2 import StudentMoEStage2


def build_task(task_name, level):
    if task_name == "lane_keeping":
        return LaneKeepingTask(curriculum_level=level, auto_curriculum=False)
    if task_name == "pedestrian":
        return PedestrianTask(curriculum_level=level, auto_curriculum=False)
    if task_name == "change_lane":
        return ChangeLaneTask(curriculum_level=level, auto_curriculum=False)
    raise ValueError(f"Unknown task: {task_name}")


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


def to_batch(x):
    if x is None:
        return None
    return np.expand_dims(np.asarray(x, dtype=np.float32), axis=0)


def parse_reset_output(reset_out):
    if isinstance(reset_out, (tuple, list)) and len(reset_out) == 2 and isinstance(reset_out[1], dict):
        obs, info = reset_out
    else:
        obs, info = reset_out, {}
    vision = info.get("vision", None) if isinstance(info, dict) else None
    return obs, info, vision


def parse_step_output(step_out):
    if isinstance(step_out, (tuple, list)) and len(step_out) == 5:
        next_obs, reward, terminated, truncated, info = step_out
        return next_obs, reward, (terminated or truncated), info
    if isinstance(step_out, (tuple, list)) and len(step_out) == 4:
        next_obs, reward, done, info = step_out
        return next_obs, reward, done, info
    raise ValueError(f"Unexpected env.step output format: {type(step_out)}")


def split_obs(obs):
    if isinstance(obs, (tuple, list)) and len(obs) == 3:
        obs_core, ego, map_state = obs
        return obs_core, ego, map_state
    return obs, None, None


def build_mask(step_idx, n_steps):
    num = min(step_idx + 1, n_steps)
    mask = np.array([1.0] * num + [0.0] * (n_steps - num), dtype=np.float32)
    return np.expand_dims(mask, axis=0)


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


def build_student(args, algo_params, observation_space, action_space):
    params = build_student_params(args, algo_params)

    if args.model_type == "stage2_moe":
        return StudentMoEStage2(params=params,
                                state_shape=observation_space.shape,
                                action_dim=action_space.high.size,
                                max_action=action_space.high[0],
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

    if args.model_type == "stage1_moe":
        return StudentMoE(params=params,
                          state_shape=observation_space.shape,
                          action_dim=action_space.high.size,
                          max_action=action_space.high[0],
                          num_experts=args.num_experts,
                          task_dim=args.task_dim,
                        )

    raise ValueError(f"Unknown model_type: {args.model_type}")


def get_student_action(student, obs_core, map_state, vision, step_idx, n_steps_mask, return_aux=False):
    mask = build_mask(step_idx, n_steps_mask)

    out = student(
        to_batch(obs_core),
        mask=tf.convert_to_tensor(mask, dtype=tf.float32),
        map_state=None if map_state is None else tf.convert_to_tensor(to_batch(map_state), dtype=tf.float32),
        vision=None if vision is None else tf.convert_to_tensor(to_batch(vision), dtype=tf.float32),
        training=False,
        return_aux=return_aux,
    )

    if return_aux:
        action = out["action"].numpy()
        if action.ndim > 1 and action.shape[0] == 1:
            action = action[0]

        aux = {
            "task_embedding": out["task_embedding"].numpy()[0].astype(np.float32),
            "gate_probs": out["gate_probs"].numpy()[0].astype(np.float32),
            "top_expert": int(np.argmax(out["gate_probs"].numpy()[0])),
        }

        if "expert_outs" in out:
            expert_outs = out["expert_outs"]
            if isinstance(expert_outs, (list, tuple)):
                aux["expert_outs"] = [e.numpy()[0].astype(np.float32) for e in expert_outs]
            else:
                aux["expert_outs"] = expert_outs.numpy()[0].astype(np.float32)

        return np.asarray(action, dtype=np.float32), aux

    action = out.numpy()
    if action.ndim > 1 and action.shape[0] == 1:
        action = action[0]
    return np.asarray(action, dtype=np.float32), None


def evaluate_checkpoint(student, env, task_name, n_episodes=20, max_episode_steps=2000, n_steps_mask=10):
    rows = []
    success_count = 0
    step_rows = []
    expert_usage_counts = None

    collision_count = 0
    stagnation_count = 0
    episode_mean_speeds = []

    for ep in range(n_episodes):
        ep_gate_probs = []
        ep_top_experts = []
        ep_task_embeddings = []

        reset_out = env.reset()
        obs, reset_info, vision = parse_reset_output(reset_out)
        obs_core, ego, map_state = split_obs(obs)

        ep_return = 0.0
        done = False
        step_idx = 0
        last_info = {}

        speed_trace = []
        low_speed_steps = 0

        while not done and step_idx < max_episode_steps:
            action, aux = get_student_action(student,
                                             obs_core,
                                             map_state,
                                             vision,
                                             step_idx,
                                             n_steps_mask,
                                             return_aux=True,
                                            )
            next_obs, reward, done, info = parse_step_output(env.step(action))

            next_vision = info.get("vision", None) if isinstance(info, dict) else None
            next_obs_core, next_ego, next_map_state = split_obs(next_obs)

            ep_return += float(reward)
            obs_core = next_obs_core
            ego = next_ego
            map_state = next_map_state
            vision = next_vision
            last_info = info if isinstance(info, dict) else {}

            if aux is not None:
                gate_probs = aux["gate_probs"]
                top_expert = aux["top_expert"]
                task_embedding = aux["task_embedding"]

                if expert_usage_counts is None:
                    expert_usage_counts = np.zeros((len(gate_probs),), dtype=np.int32)

                expert_usage_counts[top_expert] += 1
                ep_gate_probs.append(gate_probs)
                ep_top_experts.append(top_expert)
                ep_task_embeddings.append(task_embedding)

                step_row = {
                    "task_name": task_name,
                    "episode": ep,
                    "step": step_idx,
                    "top_expert": int(top_expert),
                    "speed_kmh": float(last_info.get("speed_kmh", 0.0)),
                    "progress": float(last_info.get("progress", 0.0)),
                }

                for k in range(len(gate_probs)):
                    step_row[f"gate_prob_expert_{k}"] = float(gate_probs[k])

                for d in range(len(task_embedding)):
                    step_row[f"task_emb_{d}"] = float(task_embedding[d])

                step_rows.append(step_row)

            speed_now = float(last_info.get("speed_kmh", 0.0))
            speed_trace.append(speed_now)

            if speed_now < 2.0:
                low_speed_steps += 1
            else:
                low_speed_steps = 0

            step_idx += 1

        success = bool(last_info.get("finish", False))
        success_count += int(success)

        done_reason = str(last_info.get("done_reason", "")).lower()

        is_collision = ("collision" in done_reason) or ("crash" in done_reason)
        if is_collision:
            collision_count += 1

        # duas formas de detectar estagnação:
        # 1) done_reason indicando stuck / timeout
        # 2) muitos passos seguidos com velocidade muito baixa
        is_stagnation = (("stuck" in done_reason) or 
                         ("timeout" in done_reason) or 
                         (low_speed_steps >= 50)
                        )

        if is_stagnation:
            stagnation_count += 1

        ep_mean_speed = float(np.mean(speed_trace)) if speed_trace else 0.0
        episode_mean_speeds.append(ep_mean_speed)

        if ep_gate_probs:
            ep_gate_probs_np = np.asarray(ep_gate_probs, dtype=np.float32)
            ep_mean_gate = ep_gate_probs_np.mean(axis=0)
            ep_top_experts_np = np.asarray(ep_top_experts, dtype=np.int32)
        else:
            ep_mean_gate = None
            ep_top_experts_np = None

        row = {"task_name": task_name,
               "episode": ep,
               "success": int(success),
               "return": float(ep_return),
               "length": int(step_idx),
               "done_reason": last_info.get("done_reason", None),
               "progress": float(last_info.get("progress", 0.0)),
               "dist_to_goal": float(last_info.get("dist_to_goal", 0.0)),
               "lateral_error": float(last_info.get("lateral_error", 0.0)),
               "heading_error": float(last_info.get("heading_error", 0.0)),
               "speed_kmh": float(last_info.get("speed_kmh", 0.0)),
               "collision": int(is_collision),
               "stagnation": int(is_stagnation),
               "mean_speed_kmh": ep_mean_speed,
               "dominant_expert": int(np.bincount(ep_top_experts_np).argmax()) if ep_top_experts_np is not None and len(ep_top_experts_np) > 0 else -1,
              }
        rows.append(row)

        if ep_mean_gate is not None:
            for k in range(len(ep_mean_gate)):
                row[f"mean_gate_prob_expert_{k}"] = float(ep_mean_gate[k])

        print(f"[{task_name}] ep={ep:03d} | success={success} | "
              f"return={ep_return:.3f} | steps={step_idx} | "
              f"progress={row['progress']:.3f}"
             )
        
        
    expert_usage_rate = None
    if expert_usage_counts is not None and expert_usage_counts.sum() > 0:
        expert_usage_rate = expert_usage_counts.astype(np.float32) / float(expert_usage_counts.sum())

    summary = {"task_name": task_name,
               "episodes": int(n_episodes),
               "success_rate": float(success_count / max(n_episodes, 1)),
               "collision_rate": float(collision_count / max(n_episodes, 1)),
               "stagnation_rate": float(stagnation_count / max(n_episodes, 1)),
               "mean_return": float(np.mean([r["return"] for r in rows])) if rows else 0.0,
               "mean_length": float(np.mean([r["length"] for r in rows])) if rows else 0.0,
               "mean_progress": float(np.mean([r["progress"] for r in rows])) if rows else 0.0,
               "mean_speed_kmh": float(np.mean(episode_mean_speeds)) if episode_mean_speeds else 0.0,
               "num_experts": int(len(expert_usage_counts)) if expert_usage_counts is not None else 0,
              }
    
    if expert_usage_rate is not None:
        for k in range(len(expert_usage_rate)):
            summary[f"expert_usage_rate_{k}"] = float(expert_usage_rate[k])

    
    return rows, step_rows, summary


def save_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = get_argument()
    parser.add_argument("--model-type", choices=["stage1_moe", "stage2_moe"], default="stage2_moe")
    parser.add_argument("--ckpt-dir", type=str, required=True)
    parser.add_argument("--ckpt-ids", nargs="+", required=True)
    parser.add_argument("--task", choices=["lane_keeping", "change_lane", "pedestrian"], required=True)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-outdir", type=str, default="student_eval_results")

    parser.add_argument("--task-dim", type=int, default=16)
    parser.add_argument("--geo-dim", type=int, default=8)
    parser.add_argument("--int-dim", type=int, default=8)

    parser.add_argument("--geo-type", type=str, default="mlp", choices=["mlp", "cross_attn"])
    parser.add_argument("--interaction-type", type=str, default="mlp", choices=["mlp", "cross_attn"])

    parser.add_argument("--num-experts", type=int, default=2)
    parser.add_argument("--num-old-experts", type=int, default=3)
    parser.add_argument("--num-new-experts", type=int, default=0)

    parser.add_argument("--student-use-vision", action="store_true")
    parser.add_argument("--student-vision-dim", type=int, default=280)
    parser.add_argument("--student-fusion-type", choices=["cross", "self"], default="cross")

    parser.add_argument("--use-geo", action="store_true")
    parser.add_argument("--use-int", action="store_true")

    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--traffic_manager_port", "--traffic-manager-port", dest="traffic_manager_port", type=int, default=8000,)



    # parser.add_argument("--task-dim", type=int, default=16)
    # parser.add_argument("--geo-dim", type=int, default=8)
    # parser.add_argument("--num-experts", type=int, default=2)
    # parser.add_argument("--int-dim", type=int, default=8)



    # parser.add_argument("--num-old-experts", type=int, default=3)



    # parser.add_argument("--num-new-experts", type=int, default=0)
    # parser.add_argument("--student-use-vision", action="store_true")
    # parser.add_argument("--student-vision-dim", type=int, default=280)
    # parser.add_argument("--student-fusion-type", choices=["cross", "self"], default="cross")

    args = parser.parse_args()
    args, algo_params, runner_params = set_configs(args, test=True)

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    if args.gpu is not None and args.gpu >= 0 and len(gpus) > args.gpu:
        tf.config.set_visible_devices([gpus[args.gpu]], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[args.gpu], True)
        print(gpus[args.gpu])
    else:
        tf.config.set_visible_devices([], 'GPU')

    observation_space = gym.spaces.Box(
        low=-1000, high=1000,
        shape=(args.neighbors + 1, args.N_steps, args.dim),
        dtype=np.float32,
    )
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    os.makedirs(args.eval_outdir, exist_ok=True)

    all_rows = []
    summary_rows = []
    all_step_rows = []

    task = build_task(args.task, level=args.level)
    cfg = EnvConfig()
    cfg.host = args.host
    cfg.port = args.port
    cfg.traffic_manager_port = args.traffic_manager_port

    env = CarlaRouteEnv(cfg, task=task)



    env.observation_space = observation_space
    env.action_space = action_space

    try:
        for ckpt_id in args.ckpt_ids:
            ckpt_path = os.path.join(args.ckpt_dir, ckpt_id)

            student = build_student(args, algo_params, observation_space, action_space)
            restore_student(student, ckpt_path)

            print(f"\n========== EVALUATING {ckpt_id} on {args.task} ==========")
            ep_rows, ep_step_rows, summary = evaluate_checkpoint(student=student,
                                                   env=env,
                                                   task_name=args.task,
                                                   n_episodes=args.eval_episodes,
                                                   max_episode_steps=args.episode_max_steps,
                                                   n_steps_mask=args.N_steps,
                                                  )

            for r in ep_rows:
                r["checkpoint"] = ckpt_id
                all_rows.append(r)

            for r in ep_step_rows:
                r["checkpoint"] = ckpt_id
                all_step_rows.append(r)

            summary["checkpoint"] = ckpt_id
            summary_rows.append(summary)

            print(f"[{ckpt_id}] {args.task} success = {summary['success_rate']:.3f}")
    finally:
        try:
            env.close()
        except Exception:
            pass

    save_csv(os.path.join(args.eval_outdir, f"episode_results_{args.task}.csv"), all_rows)
    save_csv(os.path.join(args.eval_outdir, f"summary_results_{args.task}.csv"), summary_rows)
    save_csv(os.path.join(args.eval_outdir, f"step_results_{args.task}.csv"), all_step_rows)

    with open(os.path.join(args.eval_outdir, f"summary_results_{args.task}.json"), "w") as f:
        json.dump(summary_rows, f, indent=2)

    print("\nSaved evaluation to:", args.eval_outdir)


if __name__ == "__main__":
    main()


# python evaluate_student_single_task_carla.py \
#   --model-type stage2_moe \
#   --ckpt-dir ../checkpoints/stage2_moe_ped_phase2_curve_taskmode_ctr \
#   --ckpt-ids ckpt-14 \
#   --task lane_keeping \
#   --eval-episodes 5 \
#   --eval-outdir eval_lane_ckpt14 \
#   --task-dim 16 \
#   --mode-dim 8 \
#   --num-old-experts 3 \
#   --num-new-experts 0 \
#   --level 3


# python evaluate_student_single_task_carla.py \
#   --model-type stage2_moe \
#   --ckpt-dir ../checkpoints/stage2_moe_ped_phase2_curve_taskmode_ctr \
#   --ckpt-ids ckpt-14 \
#   --task pedestrian \
#   --eval-episodes 5 \
#   --eval-outdir eval_ped_ckpt14 \
#   --task-dim 16 \
#   --mode-dim 8 \
#   --num-old-experts 3 \
#   --num-new-experts 0 \
#   --level 3
###########################################################################
#Phase1
# python evaluate_student_single_task_carla.py \
#   --model-type stage2_moe \
#   --ckpt-dir ../checkpoints_zgeo_zint/stage2_moe_ped_phase1_geo_int \
#   --ckpt-ids ckpt-10 \
#   --task lane_keeping \
#   --eval-episodes 50 \
#   --eval-outdir eval_phase1_lane \
#   --task-dim 16 \
#   --geo-dim 8 \
#   --int-dim 8 \
#   --geo-type cross_attn \
#   --interaction-type cross_attn \
#   --num-old-experts 2 \
#   --num-new-experts 1 \
#   --level 3


# python evaluate_student_single_task_carla.py \
#   --model-type stage2_moe \
#   --ckpt-dir ../checkpoints_zgeo_zint/stage2_moe_ped_phase1_geo_int \
#   --ckpt-ids ckpt-10 \
#   --task pedestrian \
#   --eval-episodes 50 \
#   --eval-outdir eval_phase1_ped \
#   --task-dim 16 \
#   --geo-dim 8 \
#   --int-dim 8 \
#   --geo-type cross_attn \
#   --interaction-type cross_attn \
#   --num-old-experts 2 \
#   --num-new-experts 1 \
#   --level 3
##################################################################################
# Phase2
# python evaluate_student_single_task_carla.py \
#   --model-type stage2_moe \
#   --ckpt-dir ../checkpoints_zgeo_zint/stage2_moe_ped_phase2_geo_int \
#   --ckpt-ids ckpt-20 \
#   --task pedestrian \
#   --eval-episodes 50 \
#   --eval-outdir eval_ped_ckpt20 \
#   --task-dim 16 \
#   --geo-dim 8 \
#   --int-dim 8 \
#   --geo-type cross_attn \
#   --interaction-type cross_attn \
#   --num-old-experts 3 \
#   --num-new-experts 0 \
#   --level 3
#######################################################################################################################################################
# STAGE1
#########################################################################################################################################################
# python evaluate_student_single_task_carla.py \
#   --model-type stage1_moe \
#   --ckpt-dir ../checkpoints_stage1_multitask/stage1_moe_lane_change \
#   --ckpt-ids ckpt-10 ckpt-15 ckpt-20 \
#   --task lane_keeping \
#   --eval-episodes 50 \
#   --eval-outdir eval_stage1_lane \
#   --task-dim 16 \
#   --num-experts 2 \
#   --level 1


# python evaluate_student_single_task_carla.py \
#   --model-type stage1_moe \
#   --ckpt-dir ../checkpoints_stage1_multitask/stage1_moe_lane_change \
#   --ckpt-ids ckpt-10 ckpt-15 ckpt-20 \
#   --task change_lane \
#   --eval-episodes 50 \
#   --eval-outdir eval_stage1_change \
#   --task-dim 16 \
#   --num-experts 2 \
#   --level 3
########################################################################################################################################################
# STAGE2 PHASE1 WITH GEO AND INT
########################################################################################################################################################
# python evaluate_student_single_task_carla.py \
#   --model-type stage2_moe \
#   --ckpt-dir ../checkpoints_stage2_multitask/stage2_moe_ped_phase1_simple \
#   --ckpt-ids ckpt-10 ckpt-12 ckpt-14 ckpt-16 ckpt-18 ckpt-20 \
#   --task pedestrian \
#   --eval-episodes 50 \
#   --eval-outdir eval_stage2_phase1_ped_multitask \
#   --task-dim 16 \
#   --num-old-experts 2 \
#   --num-new-experts 1 \
#   --level 3


# python evaluate_student_single_task_carla.py \
#   --model-type stage2_moe \
#   --ckpt-dir checkpoints_stage2_multitask/stage2_moe_ped_phase1_simple \
#   --ckpt-ids ckpt-16 \
#   --task pedestrian \
#   --eval-episodes 50 \
#   --eval-outdir eval_stage2_phase1_ped_ckpt16 \
#   --task-dim 16 \
#   --num-old-experts 2 \
#   --num-new-experts 1 \
#   --use-geo \
#   --use-int \
#   --level 3
