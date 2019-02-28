import argparse
import gym
import os
import datetime
import json

from RL_implementations.implementations.algorithms import TD3
from RL_implementations.implementations.algorithms import DDPG
from RL_implementations.implementations.utils import replay_buffer

import gym_multi_dimensional
import numpy as np

def populate_output_dir(path, exist):
    if exist is False:
        os.makedirs(path)

    os.makedirs(path + "/models")
    os.makedirs(path + "/visualizations")
    os.makedirs(path + "/evaluations")


def setup_output_dir(path):
    exist = os.path.exists(path)

    if exist:

        if os.path.isdir(path) is False:
            print("Output path : {} already exist and is not a directory".format(path))
            return False

        if len(os.listdir(path)) != 0:
            print("Output directory : {} already exists and is not empty".format(path))
            return False

    populate_output_dir(path, exist)
    return True


def save_arguments(args, path):
    with open(path + '/arguments.txt', 'w') as file:
        file.write(json.dumps(args))


def learn(policy_name="DDPG",
            seed=0,
            dimensions=2,
            eval_freq=5e3,
            exploration_timesteps=1e3,
            learning_timesteps=1e4,
            buffer_size=5000,
            new_exp=True,
            expl_noise=0.1,
            batch_size=64,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            verbose=True,
            acceleration=False,
            continuous=True,
            render=True,
            save=False,
            output=str(datetime.datetime.now()),
            high_reward_value=1,
            low_reward_value=0.1,
            high_reward_count='half',
            low_reward_count='half',
            mode='deterministic'
            ):
    
    if render == False:
        import matplotlib
        matplotlib.use('Agg')

    """ Delayed import of learn_policy and visualisation module to account for headless benchmarking """
    from gym_multi_dimensional.visualization import vis_2d
    from RL_implementations import learn_policy
    
    models_path = output + "/models/"
    visualizations_path = output + "/visualizations/"
    evaluations_path = output + "/evaluations/"

    description = {
            'high_reward_value': high_reward_value,
            'low_reward_value': low_reward_value,
            'high_reward_count': high_reward_count,
            'low_reward_count': low_reward_count,
            'mode': mode
            }

    environment = gym_multi_dimensional.dynamic_register(
            n_dimensions=dimensions,env_description=description,
            continuous=continuous,acceleration=acceleration)

    replay_buffer, q_values = learn_policy.learn(policy_name=policy_name,
            policy_directory=models_path,
            evaluations_directory=evaluations_path,
            visualizations_directory=visualizations_path,
            save=save,
            seed=seed,
            environment=environment,
            eval_freq=eval_freq,
            exploration_timesteps=exploration_timesteps,
            learning_timesteps=learning_timesteps,
            buffer_size=buffer_size,
            new_exp=new_exp,
            expl_noise=expl_noise,
            batch_size=batch_size,
            discount=discount,
            tau=tau,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            policy_freq=policy_freq)

    env = gym.make(environment)

    state_dim = 1
    for dim_length in env.observation_space.shape:
        state_dim *= dim_length
    action_dim = 1
    for dim_length in env.action_space.shape:
        action_dim *= dim_length
    max_action = float(env.action_space.high[0])

    env.close()

    if dimensions != 2:
        exit()

    if policy_name == "TD3":
        policy = TD3.TD3(state_dim,action_dim,max_action)
    elif policy_name == "DDPG":
        policy = DDPG.DDPG(state_dim,action_dim,max_action)

    policy.load(policy_name + "_" + environment, models_path)

    Q_values = policy.get_Q_values(env,10)

    if acceleration:
        vis_2d.visualize_Q_arrow(Q_values, save=save, path=visualizations_path)
    else:
        vis_2d.visualize_Q_contour(Q_values, save=save, path=visualizations_path)
        vis_2d.visualize_Q_contour_time(q_values, save=save, path=visualizations_path)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name",default="DDPG")
    parser.add_argument("--seed", default=np.random.randint(10000), type=int)              #seed
    parser.add_argument("--dimensions", default=2, type=int)
    parser.add_argument("--eval_freq", default=5e3, type=float)     #how often (time steps) we evaluate
    parser.add_argument("--exploration_timesteps", default=1e3, type=int) #random steps at the beginning
    parser.add_argument("--learning_timesteps", default=1e4, type=int)
    parser.add_argument("--buffer_size", default=5000, type=int)
    parser.add_argument("--no-new-exp", dest='new_exp', action="store_false")
    parser.add_argument("--expl_noise", default=0.1, type=float)    #noise
    parser.add_argument("--batch_size", default=64, type=int)      #learning batch
    parser.add_argument("--discount", default=0.99, type=float)     #discount factor
    parser.add_argument("--tau", default=0.005, type=float)         #target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)  #noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)    #range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       #frequency of delayed policy updates
    parser.add_argument('--quiet', dest='verbose', action='store_false')
    parser.add_argument('--velocity', dest='acceleration', action='store_false')
    parser.add_argument('--acceleration', dest='acceleration', action='store_true')
    parser.add_argument('--discrete', dest='continuous', action='store_false')
    parser.add_argument('--continuous', dest='continuous', action='store_true')
    parser.add_argument('--no-render', dest='render', action='store_false')
    parser.add_argument('--save', dest='save', action='store_true')
    parser.add_argument('--output', default=str(datetime.datetime.now()), type=str)
    parser.add_argument("--high_reward_value", default=1, type=float)
    parser.add_argument("--low_reward_value", default=0.1, type=float)
    parser.add_argument("--high_reward_count", default='half')
    parser.add_argument("--low_reward_count", default='half')
    parser.add_argument("--mode", default='deterministic')

    parser.set_defaults(new_exp=True)
    parser.set_defaults(verbose=True)
    parser.set_defaults(acceleration=False)
    parser.set_defaults(continuous=True)
    parser.set_defaults(render=True)
    parser.set_defaults(save=False)

    args = parser.parse_args()
    
    if setup_output_dir(args.output) is False:
        exit()

    save_arguments(vars(args), args.output)

    learn(policy_name=args.policy_name,
            seed=args.seed,
            dimensions=args.dimensions,
            eval_freq=args.eval_freq,
            exploration_timesteps=args.exploration_timesteps,
            learning_timesteps=args.learning_timesteps,
            buffer_size=args.buffer_size,
            new_exp=args.new_exp,
            expl_noise=args.expl_noise,
            batch_size=args.batch_size,
            discount=args.discount,
            tau=args.tau,
            policy_noise=args.policy_noise,
            noise_clip=args.noise_clip,
            policy_freq=args.policy_freq,
            verbose=args.verbose,
            acceleration=args.acceleration,
            continuous=args.continuous,
            render=args.render,
            save=args.save,
            output=args.output,
            high_reward_value=args.high_reward_value,
            low_reward_value=args.low_reward_value,
            high_reward_count=args.high_reward_count,
            low_reward_count=args.low_reward_count,
            mode=args.mode
            )
