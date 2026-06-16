import os
import sys
sys.path.append('../')

import numpy as np
import tensorflow as tf
import gym

from envs.carla_route_env import CarlaRouteEnv, EnvConfig
from train.init_configs import get_argument, set_configs
from algos.sac import SAC
from envs.runners import off_policy_trainer_carla

from tasks.lane_keeping import LaneKeepingTask
from tasks.pedestrian import PedestrianTask
from tasks.change_lane import ChangeLaneTask


def build_task(task_name, level):
    if task_name == "lane_keeping":
        return LaneKeepingTask(curriculum_level=level, auto_curriculum=False,)

    elif task_name == "pedestrian":
        return PedestrianTask(curriculum_level=level, auto_curriculum=False,)
    
    elif task_name == "change_lane":
        return ChangeLaneTask(curriculum_level=level, auto_curriculum=False,)

    else:
        raise ValueError(f"Unknown task: {task_name}")


def main():
    parser = get_argument()
    args = parser.parse_args()

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
        # args.model_dir = f"../train/{args.task}/{args.algo}_{cfg.seed}/ckpt"
        args.model_dir = f"../train/{args.task}/SAC_1_200k_curriculum_6p"

    if args.algo.lower() != "sac":
        raise ValueError("This script is prepared for SAC.")

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


if __name__ == "__main__":
    main()


# ============================================================
# EXEMPLOS
# ============================================================

# Só avaliar
# python export_rollouts.py \
#   --task lane_keeping \
#   --level 3 \
#   --algo SAC \
#   --model-dir ../train/lane_keeping/SAC_1/ckpt-100000 \
#   --test-episodes 100

# Avaliar e salvar exatamente essas trajetórias da avaliação
# python export_rollouts.py \
#   --task lane_keeping \
#   --level 3 \
#   --algo SAC \
#   --model-dir ../train/lane_keeping/SAC_1/ckpt-100000 \
#   --test-episodes 500 \
#   --save-eval-rollouts \
#  --save-only-success \
#   --eval-rollout-dir datasets/lane_keeping_eval

# Pedestrian
# python export_rollouts.py \
#   --task pedestrian \
#   --level 3 \
#   --algo SAC \
#   --model-dir ../train/pedestrian/SAC_1/ckpt-200000 \
#   --test-episodes 500 \
#   --save-eval-rollouts \
#   --eval-rollout-dir datasets/pedestrian_eval

# Change lane
# python export_rollouts.py \
#   --task change_lane \
#   --level 3 \
#   --algo SAC \
#   --model-dir ../train/change_lane/SAC_1/ckpt-200000 \
#   --test-episodes 500 \
#   --save-eval-rollouts \
#   --save-only-success \
#   --eval-rollout-dir datasets/change_lane_eval