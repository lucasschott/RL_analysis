import run_policy
import argparse
import gym

from implementations.algorithms import TD3
from implementations.algorithms import DDPG
from implementations.utils import replay_buffer

import gym_multi_dimensional
from gym_multi_dimensional.visualization import vis_2d

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name",default="Random")
    parser.add_argument("--policy_directory", default="policies")
    parser.add_argument("--dimensions", default=2, type=int)
    parser.add_argument("--max_episodes", default=50, type=int)
    parser.add_argument("--buffer_size", default=5000, type=int)
    parser.add_argument('--quiet', dest='verbose', action='store_false')
    parser.set_defaults(verbose=True)
    parser.add_argument('--velocity', dest='acceleration', action='store_false')
    parser.add_argument('--acceleration', dest='acceleration', action='store_true')
    parser.set_defaults(acceleration=False)
    parser.add_argument('--discrete', dest='continuous', action='store_false')
    parser.add_argument('--continuous', dest='continuous', action='store_true')
    parser.set_defaults(continuous=True)
    parser.add_argument('--no-render', dest='render', action='store_false')
    parser.set_defaults(render=True)

    args = parser.parse_args()

    environment = gym_multi_dimensional.dynamic_register(n_dimensions=args.dimensions,
            env_description={},continuous=args.continuous,acceleration=args.acceleration)

    replay_buffer = run_policy.run_policy(policy_name=args.policy_name,
            policy_directory=args.policy_directory,
            environment=environment,
            max_episodes=args.max_episodes,
            buffer_size=args.buffer_size,
            render=args.render,
            verbose=args.verbose)

    #vis_2d.visualize_RB(replay_buffer)
