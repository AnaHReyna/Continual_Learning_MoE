import os
import sys
sys.path.append('../../')

import csv
import json

import numpy as np
import tensorflow as tf
import gym

from envs.carla_route_env import CarlaRouteEnv, EnvConfig
from train.init_configs import get_argument, set_configs
from tasks.lane_keeping import LaneKeepingTask
from tasks.pedestrian import PedestrianTask

from student_model_moe import StudentMoE
from student_model_moe_stage2 import StudentMoEStage2


def build_task(task_name, level):
    if task_name == "lane_keeping":
        return LaneKeepingTask(curriculum_level=level, auto_curriculum=False)
    if task_name == "pedestrian":
        return PedestrianTask(curriculum_level=level, auto_curriculum=False)
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
                                mode_dim=args.mode_dim,
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


def get_student_action(student, obs_core, map_state, vision, step_idx, n_steps_mask):
    mask = build_mask(step_idx, n_steps_mask)
    out = student(to_batch(obs_core),
                  mask=tf.convert_to_tensor(mask, dtype=tf.float32),
                  map_state=None if map_state is None else tf.convert_to_tensor(to_batch(map_state), dtype=tf.float32),
                  vision=None if vision is None else tf.convert_to_tensor(to_batch(vision), dtype=tf.float32),
                  training=False,
                  return_aux=False,
                 )
    
    action = out.numpy()
    if action.ndim > 1 and action.shape[0] == 1:
        action = action[0]
    return np.asarray(action, dtype=np.float32)


def evaluate_checkpoint(student, env, task_name, n_episodes=20, max_episode_steps=2000, n_steps_mask=10):
    rows = []
    success_count = 0

    for ep in range(n_episodes):
        reset_out = env.reset()
        obs, reset_info, vision = parse_reset_output(reset_out)
        obs_core, ego, map_state = split_obs(obs)

        ep_return = 0.0
        done = False
        step_idx = 0
        last_info = {}

        while not done and step_idx < max_episode_steps:
            action = get_student_action(student, obs_core, map_state, vision, step_idx, n_steps_mask)
            next_obs, reward, done, info = parse_step_output(env.step(action))

            next_vision = info.get("vision", None) if isinstance(info, dict) else None
            next_obs_core, next_ego, next_map_state = split_obs(next_obs)

            ep_return += float(reward)
            obs_core = next_obs_core
            ego = next_ego
            map_state = next_map_state
            vision = next_vision
            last_info = info if isinstance(info, dict) else {}
            step_idx += 1

        success = bool(last_info.get("finish", False))
        success_count += int(success)

        rows.append({"task_name": task_name,
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
                    }
                    )

        print(f"[{task_name}] ep={ep:03d} | success={success} | "
              f"return={ep_return:.3f} | steps={step_idx} | "
              f"progress={rows[-1]['progress']:.3f}"
             )

    summary = {"task_name": task_name,
               "episodes": int(n_episodes),
               "success_rate": float(success_count / max(n_episodes, 1)),
               "mean_return": float(np.mean([r["return"] for r in rows])) if rows else 0.0,
               "mean_length": float(np.mean([r["length"] for r in rows])) if rows else 0.0,
               "mean_progress": float(np.mean([r["progress"] for r in rows])) if rows else 0.0,
              }
    
    return rows, summary


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
    parser.add_argument("--tasks", nargs="+", default=["lane_keeping", "pedestrian"])
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-outdir", type=str, default="student_eval_results")
    parser.add_argument("--task-dim", type=int, default=16)
    parser.add_argument("--mode-dim", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=2)
    parser.add_argument("--num-old-experts", type=int, default=3)
    parser.add_argument("--num-new-experts", type=int, default=0)
    parser.add_argument("--student-use-vision", action="store_true")
    parser.add_argument("--student-vision-dim", type=int, default=280)
    parser.add_argument("--student-fusion-type", choices=["cross", "self"], default="cross")

    args = parser.parse_args()
    args, algo_params, runner_params = set_configs(args, test=True)

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    if args.gpu is not None and args.gpu >= 0 and len(gpus) > args.gpu:
        tf.config.set_visible_devices([gpus[args.gpu]], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[args.gpu], True)
        print(gpus[args.gpu])
    else:
        tf.config.set_visible_devices([], 'GPU')

    observation_space = gym.spaces.Box(low=-1000, high=1000,
                                       shape=(args.neighbors + 1, args.N_steps, args.dim),
                                       dtype=np.float32,
                                      )
    
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    os.makedirs(args.eval_outdir, exist_ok=True)

    all_rows = []
    summary_rows = []

    for ckpt_id in args.ckpt_ids:
        ckpt_path = os.path.join(args.ckpt_dir, ckpt_id)

        student = build_student(args, algo_params, observation_space, action_space)
        restore_student(student, ckpt_path)

        ckpt_task_success = {}

        for task_name in args.tasks:
            task = build_task(task_name, level=args.level)
            cfg = EnvConfig()
            env = CarlaRouteEnv(cfg, task=task)
            env.observation_space = observation_space
            env.action_space = action_space

            print(f"\n========== EVALUATING {ckpt_id} on {task_name} ==========")
            ep_rows, summary = evaluate_checkpoint(student=student,
                                                   env=env,
                                                   task_name=task_name,
                                                   n_episodes=args.eval_episodes,
                                                   max_episode_steps=args.episode_max_steps,
                                                   n_steps_mask=args.N_steps,
                                                  )

            for r in ep_rows:
                r["checkpoint"] = ckpt_id
                all_rows.append(r)

            summary["checkpoint"] = ckpt_id
            summary_rows.append(summary)
            ckpt_task_success[task_name] = summary["success_rate"]

            try:
                env.close()
            except Exception:
                pass

        avg_success = float(np.mean(list(ckpt_task_success.values()))) if ckpt_task_success else 0.0
        summary_rows.append({"checkpoint": ckpt_id,
                             "task_name": "AVG",
                             "episodes": 0,
                             "success_rate": avg_success,
                             "mean_return": 0.0,
                             "mean_length": 0.0,
                             "mean_progress": 0.0,
                            }
                           )
        
        print(f"\n[{ckpt_id}] AVG success = {avg_success:.3f}")

    save_csv(os.path.join(args.eval_outdir, "episode_results.csv"), all_rows)
    save_csv(os.path.join(args.eval_outdir, "summary_results.csv"), summary_rows)

    with open(os.path.join(args.eval_outdir, "summary_results.json"), "w") as f:
        json.dump(summary_rows, f, indent=2)

    print("\nSaved evaluation to:", args.eval_outdir)


if __name__ == "__main__":
    main()



# Exemplo para um teste curto do Stage 2 MoE:
# python evaluate_student_ckpts_carla.py \
#   --model-type stage2_moe \
#   --ckpt-dir ../checkpoints/stage2_moe_ped_phase2_curve_taskmode_ctr \
#   --ckpt-ids ckpt-14 \
#   --tasks lane_keeping pedestrian \
#   --eval-episodes 5 \
#   --eval-outdir eval_smoke_test \
#   --task-dim 16 \
#   --mode-dim 8 \
#   --num-old-experts 3 \
#   --num-new-experts 0