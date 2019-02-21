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
    parser.add_argument("--max_episodes", default=100, type=int)
    parser.add_argument("--buffer_size", default=10000, type=int)
    parser.add_argument("--batch_size", default=1000, type=int)
    parser.add_argument('--quiet', dest='verbose', action='store_false')
    parser.set_defaults(verbose=True)
    parser.add_argument('--velocity', dest='acceleration', action='store_false')
    parser.add_argument('--acceleration', dest='acceleration', action='store_true')
    parser.set_defaults(acceleration=True)
    parser.add_argument('--discrete', dest='continuous', action='store_false')
    parser.add_argument('--continuous', dest='continuous', action='store_true')
    parser.set_defaults(continuous=True)
    parser.add_argument('--no-render', dest='render', action='store_false')
    parser.set_defaults(render=True)

    args = parser.parse_args()

    environment = gym_multi_dimensional.dynamic_register(n_dimensions=args.dimensions,
            env_description={},continuous=args.continuous,acceleration=args.acceleration)

    print("run policy")

    rb = run_policy.run_policy(policy_name="Random",
            policy_directory=args.policy_directory,
            environment=environment,
            max_episodes=args.max_episodes,
            buffer_size=args.buffer_size,
            render=args.render,
            verbose=args.verbose)

    print("uniform sample")

    sample = rb.uniform_sample(args.batch_size)
    rb_sample = replay_buffer.ReplayBuffer(args.batch_size,sample)
    #replay buffer filter

    print("visualize RB")

    vis_2d.visualize_RB(rb)

    env = gym.make(environment)

    state_dim = 1
    for dim_length in env.observation_space.shape:
        state_dim *= dim_length
    action_dim = 1
    for dim_length in env.action_space.shape:
        action_dim *= dim_length
    max_action = float(env.action_space.high[0])

    env.close()

    if args.policy_name == "TD3":
        policy = TD3.TD3(state_dim,action_dim,max_action)
    elif args.policy_name == "DDPG":
        policy = DDPG.DDPG(state_dim,action_dim,max_action)

    policy.load(args.policy_name + "_" + environment,"policies")

    print("compute Q values")

    Q_values = policy.Q_values(rb_sample)

    print("visualize Q")

    vis_2d.visualize_Q_arrow(Q_values)
    vis_2d.visualize_Q_contour(Q_values)
