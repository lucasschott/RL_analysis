import learn_policy
import argparse
import gym

from implementations.algorithms import TD3
from implementations.algorithms import DDPG
from implementations.utils import replay_buffer

import gym_multi_dimensional
from gym_multi_dimensional.visualization import vis_2d
import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name",default="DDPG")
    parser.add_argument("--policy_directory", default="policies")
    parser.add_argument("--seed", default=np.random.randint(10000), type=int)              #seed
    parser.add_argument("--dimensions", default=2, type=int)
    parser.add_argument("--eval_freq", default=5e3, type=float)     #how often (time steps) we evaluate
    parser.add_argument("--start_timesteps", default=1e3, type=int) #random steps at the beginning
    parser.add_argument("--max_timesteps", default=1e4, type=int)
    parser.add_argument("--buffer_size", default=5000, type=int)
    parser.add_argument("--no-new-exp", dest='new_exp', action="store_false")
    parser.set_defaults(new_exp=True)
    parser.add_argument("--expl_noise", default=0.1, type=float)    #noise
    parser.add_argument("--batch_size", default=64, type=int)      #learning batch
    parser.add_argument("--discount", default=0.99, type=float)     #discount factor
    parser.add_argument("--tau", default=0.005, type=float)         #target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)  #noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)    #range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       #frequency of delayed policy updates
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


    replay_buffer, q_values = learn_policy.learn_policy(policy_name=args.policy_name,
            policy_directory=args.policy_directory,
            seed=args.seed,
            environment=environment,
            eval_freq=args.eval_freq,
            start_timesteps=args.start_timesteps,
            max_timesteps=args.max_timesteps,
            buffer_size=args.buffer_size,
            new_exp=args.new_exp,
            expl_noise=args.expl_noise,
            batch_size=args.batch_size,
            discount=args.discount,
            tau=args.tau,
            policy_noise=args.policy_noise,
            noise_clip=args.noise_clip,
            policy_freq=args.policy_freq)

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

    Q_values = policy.get_Q_values(env,10)
    if args.acceleration:
        vis_2d.visualize_Q_arrow(Q_values)
    else:
        vis_2d.visualize_Q_contour(Q_values)
        vis_2d.visualize_Q_contour_time(q_values, args.policy_name)

