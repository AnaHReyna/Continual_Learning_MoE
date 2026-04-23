import sys
sys.path.append('../')
import tensorflow as tf
import gym
from envs.carla_route_env import CarlaRouteEnv, EnvConfig
from train.init_configs import get_argument, set_configs      
from envs.runners import on_policy_trainer_carla, off_policy_trainer_carla


from tasks.lane_keeping import LaneKeepingTask
from tasks.pedestrian import PedestrianTask

args = get_argument().parse_args()
args, algo_params, runner_params = set_configs(args, test=True)


def build_task(task_name, level):
    if task_name == "lane_keeping":
        return LaneKeepingTask(curriculum_level=level,
                               auto_curriculum=False,
                               )
    
    elif task_name == "pedestrian":
        return PedestrianTask(curriculum_level=level,
                              auto_curriculum=False,
                              )
    
    else:
        raise ValueError(f"Unknown task: {task_name}")
    

def main():
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    if args.gpu is not None and args.gpu >= 0 and len(gpus) > args.gpu:
        tf.config.set_visible_devices([gpus[args.gpu]], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[args.gpu], True)
        print(gpus[args.gpu])
    else:
        tf.config.set_visible_devices([], 'GPU')

    task = build_task(args.task, args.level)

    OBSERVATION_SPACE = gym.spaces.Box(low=-1000, high=1000, shape=(args.neighbors + 1, args.N_steps, args.dim,))
    ACTION_SPACE = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))

    cfg = EnvConfig()
    test_env = CarlaRouteEnv(cfg, task=task)
    test_env.observation_space = OBSERVATION_SPACE
    test_env.action_space = ACTION_SPACE

    if args.model_dir == None:
        args.model_dir = f"../train/{args.task}/{args.algo}_{cfg.seed}/ckpt"


    if args.algo == 'ppo':
        from algos.modules.actor_critic_policy import GaussianActorCritic
        from algos.ppo import PPO
        policy = PPO(actor_critic=GaussianActorCritic(**algo_params['actor_critic_params']), **algo_params)
        runner_page = on_policy_trainer_carla
        runner = runner_page.OnPolicyTrainer(policy=policy, env=test_env, args=args, test_env=test_env, **runner_params)
        runner.evaluate_policy()

    else:
        from algos.sac import SAC
        policy = SAC(state_shape=OBSERVATION_SPACE.shape, 
                     action_dim=ACTION_SPACE.high.size, 
                     max_action=ACTION_SPACE.high[0], 
                     **algo_params
                     )
        
        runner_page = off_policy_trainer_carla
        runner = runner_page.Trainer(policy=policy, 
                                     env=test_env, 
                                     test_env=test_env, 
                                     args=args, 
                                     **runner_params
                                     )
        
        runner.evaluate_policy_continuously()
        


if __name__ == "__main__":
    main()
