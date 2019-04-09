import argparse
import gym

from RL_implementations.implementations.algorithms import TD3
from RL_implementations.implementations.algorithms import DDPG

import gym_hypercube

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name",default="DDPG")
    parser.add_argument("--policy_directory", default="models")
    parser.add_argument("--dimensions", default=2, type=int)
    parser.set_defaults(verbose=True)
    parser.add_argument('--velocity', dest='acceleration', action='store_false')
    parser.add_argument('--acceleration', dest='acceleration', action='store_true')
    parser.set_defaults(acceleration=False)
    parser.add_argument('--discrete', dest='continuous', action='store_false')
    parser.add_argument('--continuous', dest='continuous', action='store_true')
    parser.set_defaults(continuous=True)

    args = parser.parse_args()
    
    description = {
            'high_reward_value': 1,
            'low_reward_value': 0.1,
            'high_reward_count': "half",
            'low_reward_count': "half",
            'mode': "deterministic"
            }

    environment = gym_hypercube.dynamic_register(n_dimensions=args.dimensions,
            env_description=description,continuous=args.continuous,acceleration=args.acceleration)

    env = gym.make(environment)

    state_dim = 1
    for dim_length in env.observation_space.shape:
        state_dim *= dim_length
    action_dim = 1
    for dim_length in env.action_space.shape:
        action_dim *= dim_length
    max_action = float(env.action_space.high[0])


    if args.policy_name == "TD3":
        policy = TD3.TD3(state_dim,action_dim,max_action)
    elif args.policy_name == "DDPG":
        policy = DDPG.DDPG(state_dim,action_dim,max_action)

    policy.load(args.policy_name + "_" + environment,args.policy_directory)

    print("Q values")

    Q_values = policy.get_Q_values(env,3)

    print(Q_values)

    env.close()
