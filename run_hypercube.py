from  RL_implementations import run_policy
import argparse
import gym

from RL_implementations.implementations.algorithms import TD3
from RL_implementations.implementations.algorithms import DDPG
from RL_implementations.implementations.utils import replay_buffer

import gym_hypercube
from gym_hypercube.visualization import vis_2d

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm",default="Random")
    parser.add_argument("--policy_directory", default="results/models")
    parser.add_argument("--dimensions", default=2, type=int)
    parser.add_argument("--max_episodes", default=50, type=int)
    parser.add_argument("--max_timesteps", default=1e4, type=int)
    parser.add_argument("--buffer_size", default=5000, type=int)
    parser.add_argument('--quiet', dest='verbose', action='store_false')
    parser.add_argument('--acceleration', dest='acceleration', action='store_true')
    parser.add_argument('--discrete', dest='discrete', action='store_true')
    parser.add_argument('--no_render', dest='render', action='store_false')
    parser.add_argument("--high_reward_value", default=1, type=float)
    parser.add_argument("--low_reward_value", default=0.1, type=float)
    parser.add_argument("--high_reward_count", default='half')
    parser.add_argument("--low_reward_count", default='half')
    parser.add_argument("--mode", default='deterministic')
    parser.add_argument("--speed_limit_mode", default='vector_norm')
    parser.add_argument("--reset_radius", default=1, type=float)
    parser.add_argument('--replay_buffer_visu', dest='replay_buffer_visu', action='store_true') #visualize replay buffer

    parser.set_defaults(verbose=True)
    parser.set_defaults(acceleration=False)
    parser.set_defaults(discrete=False)
    parser.set_defaults(render=True)
    parser.set_defaults(replay_buffer_visu=False)

    args = parser.parse_args()
    
    description = {
            'high_reward_value': args.high_reward_value,
            'low_reward_value': args.low_reward_value,
            'high_reward_count': args.high_reward_count,
            'low_reward_count': args.low_reward_count,
            'mode': args.mode,
            'speed_limit_mode': args.speed_limit_mode
            }

    environment = gym_hypercube.dynamic_register(
            n_dimensions=args.dimensions,env_description=description,
            continuous=not(args.discrete),acceleration=args.acceleration,reset_radius=args.reset_radius)

    replay_buffer = run_policy.run_policy(algorithm=args.algorithm,
            policy_directory=args.policy_directory,
            environment=environment,
            max_episodes=args.max_episodes,
            max_timesteps=args.max_timesteps,
            buffer_size=args.buffer_size,
            render=args.render,
            verbose=args.verbose)

    if args.replay_buffer_visu:
        vis_2d.visualize_RB(replay_buffer, args.acceleration, filter=None, save=False, path="")
