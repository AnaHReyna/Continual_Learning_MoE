import os
import sys
sys.path.append('../')
import pickle
import numpy as np
import tensorflow as tf
import gym

from envs.carla_route_env import CarlaRouteEnv, EnvConfig
from train.init_configs import get_argument, set_configs
from algos.sac import SAC
from envs.runners import off_policy_trainer_carla

from tasks.lane_keeping import LaneKeepingTask
from tasks.pedestrian import PedestrianTask


def build_task(task_name, level):
    if task_name == "lane_keeping":
        return LaneKeepingTask(curriculum_level=level, auto_curriculum=False,)

    elif task_name == "pedestrian":
        return PedestrianTask(
            curriculum_level=level,
            auto_curriculum=False,
        )

    else:
        raise ValueError(f"Unknown task: {task_name}")


def to_batch(x):
    if x is None:
        return None
    return np.expand_dims(np.asarray(x, dtype=np.float32), axis=0)


def parse_reset_output(reset_out):
    if isinstance(reset_out, (tuple, list)) and len(reset_out) == 2 and isinstance(reset_out[1], dict):
        obs, info = reset_out
    else:
        obs, info = reset_out, {}
    vision = info.get("vision", None)
    return obs, info, vision


def split_obs(obs):
    if isinstance(obs, (tuple, list)) and len(obs) == 3:
        obs_core, ego, map_state = obs
        return obs_core, ego, map_state
    return obs, None, None


def build_mask(step_idx, n_steps):
    num = min(step_idx + 1, n_steps)
    mask = np.array([1.0] * num + [0.0] * (n_steps - num), dtype=np.float32)
    return np.expand_dims(mask, axis=0)


def parse_step_output(step_out):
    if isinstance(step_out, (tuple, list)) and len(step_out) == 5:
        next_obs, reward, terminated, truncated, info = step_out
        done = terminated or truncated
        return next_obs, reward, done, info

    if isinstance(step_out, (tuple, list)) and len(step_out) == 4:
        next_obs, reward, done, info = step_out
        return next_obs, reward, done, info

    raise ValueError(f"Unexpected env.step output format: {type(step_out)}")


def to_numpy_action(action):
    if hasattr(action, "numpy"):
        action = action.numpy()

    action = np.asarray(action, dtype=np.float32)

    if action.ndim > 1 and action.shape[0] == 1:
        action = action[0]

    return action


def export_rollouts(policy,
                    env,
                    out_dir,
                    n_episodes=100,
                    task_name="lane_keeping",
                    skip_timestep=3,
                    n_steps_mask=10,
                    use_vision=True,
                    max_episode_steps=300):
    
    os.makedirs(out_dir, exist_ok=True)

    for ep in range(n_episodes):
        reset_out = env.reset()
        obs, reset_info, vision = parse_reset_output(reset_out)
        obs_core, ego, map_state = split_obs(obs)

        episode_return = 0.0
        done = False
        step_idx = 0
        current_action = np.zeros(env.action_space.shape, dtype=np.float32)

        traj = {"task_name": task_name,
                "episode_id": ep,
                "steps": [],
                }

        while not done and step_idx < max_episode_steps:
            mask = build_mask(step_idx, n_steps_mask)

            if step_idx % skip_timestep == 0:
                action_out = policy.get_action(obs_core,
                                               mask=mask,
                                               map_state=to_batch(map_state),
                                               test=True,
                                               vision=to_batch(vision) if use_vision else None,
                                              )
                
                current_action = to_numpy_action(action_out)

            step_out = env.step(current_action)
            next_obs, reward, done, info = parse_step_output(step_out)

            next_vision = info.get("vision", None) if isinstance(info, dict) else None
            next_obs_core, next_ego, next_map_state = split_obs(next_obs)

            traj["steps"].append({"obs": np.asarray(obs_core, dtype=np.float32),
                                  "act": np.asarray(current_action, dtype=np.float32),
                                  "rew": float(reward),
                                  "next_obs": np.asarray(next_obs_core, dtype=np.float32),
                                  "done": bool(done),
                                  "vision": None if vision is None else np.asarray(vision, dtype=np.float32),
                                  "next_vision": None if next_vision is None else np.asarray(next_vision, dtype=np.float32),
                                  "map_state": None if map_state is None else np.asarray(map_state, dtype=np.float32),
                                  "next_map_state": None if next_map_state is None else np.asarray(next_map_state, dtype=np.float32),
                                  "info": {"finish": bool(info.get("finish", False)) if isinstance(info, dict) else False,
                                           "done_reason": info.get("done_reason", None) if isinstance(info, dict) else None,
                                           "progress": float(info.get("progress", 0.0)) if isinstance(info, dict) else 0.0,
                                           "dist_to_goal": float(info.get("dist_to_goal", 0.0)) if isinstance(info, dict) else 0.0,
                                           "lateral_error": float(info.get("lateral_error", 0.0)) if isinstance(info, dict) else 0.0,
                                           "heading_error": float(info.get("heading_error", 0.0)) if isinstance(info, dict) else 0.0,
                                           "speed_kmh": float(info.get("speed_kmh", 0.0)) if isinstance(info, dict) else 0.0,
                                           },
                                  }
                                )

            episode_return += reward
            obs_core = next_obs_core
            ego = next_ego
            map_state = next_map_state
            vision = next_vision
            step_idx += 1

        traj["success"] = bool(traj["steps"][-1]["info"]["finish"]) if len(traj["steps"]) > 0 else False
        traj["return"] = float(episode_return)
        traj["length"] = int(len(traj["steps"]))

        out_path = os.path.join(out_dir,f"{task_name}_level{getattr(env.task, 'curriculum_level', 0)}_ep_{ep:04d}.pkl")

        with open(out_path, "wb") as f:
            pickle.dump(traj, f)

        print(f"[{task_name}] episódio {ep:04d} | "
              f"steps={traj['length']} | "
              f"return={traj['return']:.3f} | "
              f"success={traj['success']} | "
              f"saved={out_path}"
             )


def main():
    args = get_argument().parse_args()
    args, algo_params, runner_params = set_configs(args, test=True)

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    if args.gpu is not None and args.gpu >= 0 and len(gpus) > args.gpu:
        tf.config.set_visible_devices([gpus[args.gpu]], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[args.gpu], True)
        print(gpus[args.gpu])
    else:
        tf.config.set_visible_devices([], 'GPU')

    task = build_task(args.task, level=args.level)

    observation_space = gym.spaces.Box(low=-1000, high=1000, shape=(args.neighbors + 1, args.N_steps, args.dim), dtype=np.float32,)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32,)

    cfg = EnvConfig()
    test_env = CarlaRouteEnv(cfg, task=task)
    test_env.observation_space = observation_space
    test_env.action_space = action_space

    if args.model_dir is None:
        args.model_dir = f"../train/{args.task}/{args.algo}_{cfg.seed}/ckpt"

    if args.algo.lower() != "sac":
        raise ValueError("Este script está preparado para SAC.")

    policy = SAC(state_shape=observation_space.shape,
                 action_dim=action_space.high.size,
                 max_action=action_space.high[0],
                 **algo_params,
                )

    runner = off_policy_trainer_carla.Trainer(policy=policy,
                                              env=test_env,
                                              test_env=test_env,
                                              args=args,
                                              **runner_params
                                             )

    print("\n========== ASSESSMENT ==========")
    runner.evaluate_policy_continuously()

    logdir = args.logdir if args.logdir is not None else os.path.join("datasets", args.task)

    print("\n========== DATASET EXPORT ==========")
    export_rollouts(policy=policy,
                    env=test_env,
                    logdir=logdir,
                    n_episodes=args.test_episodes,
                    task_name=args.task,
                    skip_timestep=args.skip_timestep,
                    n_steps_mask=args.N_steps,
                    use_vision=algo_params.get("params", {}).get("use_vision", False),
                    max_episode_steps=args.episode_max_steps,
                   )


if __name__ == "__main__":
    main()


# python export_rollouts.py \
#   --task lane_keeping \
#   --level 2 \
#   --algo SAC \
#   --model-dir ../train/lane_keeping/SAC_0/ckpt-100000 \
#   --test-episodes 100 \
#   --out-dir datasets/lane_keeping


# python export_rollouts.py \
#   --task pedestrian \
#   --level 0 \
#   --algo SAC \
#   --model-dir ../train/pedestrian/SAC_0/ckpt-200000 \
#   --test-episodes 100 \
#   --out-dir datasets/pedestrian