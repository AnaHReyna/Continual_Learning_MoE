import sys
sys.path.append('../')
import tensorflow as tf
import gym
from envs.carla_route_env import CarlaRouteEnv, EnvConfig

from init_configs import get_argument, set_configs
from algos.sac import SAC
from envs.runners.off_policy_trainer_carla import Trainer

from tasks.lane_keeping import LaneKeepingTask
from tasks.pedestrian import PedestrianTask
from tasks.change_lane import ChangeLaneTask


args = get_argument().parse_args()
args, algo_params, runner_params = set_configs(args, test=False)


def build_task(task_name, level=0):
    if task_name == "lane_keeping":
        return LaneKeepingTask(curriculum_level=level,
                               auto_curriculum= True,
                               max_level=3,
                               window_size=30,
                               promote_threshold=0.65,
                               cooldown_episodes=10,
                               allow_demotion=False,
                               )
    
    elif task_name == "pedestrian":
        return PedestrianTask(curriculum_level=0,
                              auto_curriculum=True,
                              max_level=3,
                              window_size=30,
                              promote_threshold=0.65,
                              cooldown_episodes=10,
                              allow_demotion=False,
                              )
    
    elif task_name == "change_lane":
        return ChangeLaneTask(curriculum_level=0,
                              auto_curriculum=True,
                              max_level=3,
                              window_size=30,
                              promote_threshold=0.65,
                              cooldown_episodes=10,
                              allow_demotion=False,
                             )
    
    else:
        raise ValueError(f"Unknown task: {task_name}")
    


def main():
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if args.gpu is not None and args.gpu >= 0 and len(gpus) > args.gpu:
        tf.config.set_visible_devices([gpus[args.gpu]], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[args.gpu], True)
        print("Usando GPU:", gpus[args.gpu])
    else:
        tf.config.set_visible_devices([], 'GPU')


    task = build_task(args.task, args.level)

    OBSERVATION_SPACE = gym.spaces.Box(low=-1000, high=1000, shape=(args.neighbors + 1, args.N_steps, args.dim,))  
    ACTION_SPACE = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))

    cfg = EnvConfig()

    env = CarlaRouteEnv(cfg, task=task)
    env.observation_space = OBSERVATION_SPACE     
    env.action_space = ACTION_SPACE
    # args.model_dir = None

    policy = SAC(state_shape=OBSERVATION_SPACE.shape, 
                 action_dim=ACTION_SPACE.high.size, 
                 max_action=ACTION_SPACE.high[0], 
                 **algo_params
                )

    runner = Trainer(policy=policy, 
                     env=env, 
                     args=args, 
                     test_env=env, 
                     **runner_params
                    )
    runner()


if __name__ == "__main__":
    main()
