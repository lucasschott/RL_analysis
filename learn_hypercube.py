import argparse
import gym
import os
import datetime
import json

from RL_implementations.implementations.algorithms import TD3
from RL_implementations.implementations.algorithms import DDPG
from RL_implementations.implementations.utils import replay_buffer
from RL_implementations.implementations.utils import circle_filter

import gym_hypercube
import numpy as np

import matplotlib

from gym_hypercube.visualization import vis_2d
from RL_implementations import learn_policy


def learn(algorithm="DDPG",
            seed=0,
            dimensions=2,
            eval_freq=5e3,
            exploration_timesteps=1e3,
            exploration_mode="sequential",
            learning_timesteps=1e4,
            buffer_size=5000,
            new_exp=True,
            expl_noise=0.1,
            batch_size=64,
            discount=0.99,
            actor_dim=(40,30),
            critic_dim=(40,30),
            learning_rate=1e-4,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            verbose=True,
            replay_buffer_visu=False,
            policy_visu=True,
            acceleration=False,
            continuous=True,
            render=True,
            save=False,
            output="results",
            high_reward_value=1,
            low_reward_value=0.1,
            high_reward_count='half',
            low_reward_count='half',
            mode='deterministic',
            reset_radius=1,
            filter=False,
            filter_pos=0,
            filter_radius=0.2
            ):

    if render == False:
        matplotlib.use('Agg')


    description = {
            'high_reward_value': high_reward_value,
            'low_reward_value': low_reward_value,
            'high_reward_count': high_reward_count,
            'low_reward_count': low_reward_count,
            'mode': mode
            }

    if filter is True:
        assert abs(filter_pos) <= 1,  "Filter pos (%f) invalid : must be between [-1, 1]" % (filter_pos)
        filter = circle_filter.CircleFilter([filter_pos] * dimensions, filter_radius)
    else:
        filter = None

    environment = gym_hypercube.dynamic_register(
            n_dimensions=dimensions,env_description=description,
            continuous=continuous,acceleration=acceleration, reset_radius=reset_radius)

    replay_buffer, q_values , pi_values = learn_policy.learn(algorithm=algorithm,
            output=output,
            save=save,
            seed=seed,
            environment=environment,
            eval_freq=eval_freq,
            exploration_timesteps=exploration_timesteps,
            exploration_mode=exploration_mode,
            learning_timesteps=learning_timesteps,
            buffer_size=buffer_size,
            new_exp=new_exp,
            expl_noise=expl_noise,
            batch_size=batch_size,
            discount=discount,
            actor_dim=actor_dim,
            critic_dim=critic_dim,
            learning_rate=learning_rate,
            tau=tau,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            policy_freq=policy_freq,
            filter=filter,
            verbose=verbose,
            render=render)

    if dimensions==2:

        if replay_buffer_visu:
            vis_2d.visualize_RB(replay_buffer, acceleration, filter=filter, save=save, path=output+"/visualizations")

        if acceleration or not policy_visu:
            pass
        else:
            vis_2d.visualize_Q(q_values[-1], save=save, path=output+"/visualizations")
            vis_2d.visualize_Pi(pi_values[-1], save=save, path=output+"/visualizations")
            vis_2d.visualize_Q_time(q_values, save=save, path=output+"/visualizations", steps_name="timestep", steps=np.arange(0,len(q_values))*eval_freq, fps=4)
            vis_2d.visualize_Pi_time(pi_values, save=save, path=output+"/visualizations", steps_name="timestep", steps=np.arange(0,len(q_values))*eval_freq, fps=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm",default="DDPG")
    parser.add_argument("--seed", default=np.random.randint(10000), type=int)              #seed
    parser.add_argument("--dimensions", default=2, type=int)
    parser.add_argument("--eval_freq", default=2e3, type=int)     #how often (time steps) we evaluate
    parser.add_argument("--exploration_timesteps", default=1e4, type=int) #random steps at the beginning
    parser.add_argument("--exploration_mode", default="sequential", type=str)
    parser.add_argument("--learning_timesteps", default=1e4, type=int)
    parser.add_argument("--buffer_size", default=5000, type=int)
    parser.add_argument("--no_new_exp", dest='new_exp', action="store_false")
    parser.add_argument("--expl_noise", default=0.1, type=float)    #noise
    parser.add_argument("--batch_size", default=64, type=int)      #learning batch
    parser.add_argument("--discount", default=0.99, type=float)     #discount factor
    parser.add_argument("--actor_hl1", default=40, type=int) #actor hidden layer 1 size
    parser.add_argument("--actor_hl2", default=30, type=int) #actor hidden layer 2 size
    parser.add_argument("--critic_hl1", default=40, type=int) #critic hidden layer 1 size
    parser.add_argument("--critic_hl2", default=30, type=int) #critic hidden layer 2 size
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--tau", default=0.005, type=float)         #target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)  #noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)    #range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       #frequency of delayed policy updates
    parser.add_argument('--quiet', dest='verbose', action='store_false')
    parser.add_argument('--velocity', dest='acceleration', action='store_false')
    parser.add_argument('--acceleration', dest='acceleration', action='store_true')
    parser.add_argument('--discrete', dest='continuous', action='store_false')
    parser.add_argument('--continuous', dest='continuous', action='store_true')
    parser.add_argument('--replay_buffer_visu', dest='replay_buffer_visu', action='store_true') #visualize replay buffer
    parser.add_argument('--no_policy_visu', dest='policy_visu', action='store_false')
    parser.add_argument('--no_render', dest='render', action='store_false')
    parser.add_argument('--save', dest='save', action='store_true')
    parser.add_argument('--output', default="results", type=str)
    parser.add_argument("--high_reward_value", default=1, type=float)
    parser.add_argument("--low_reward_value", default=0.1, type=float)
    parser.add_argument("--high_reward_count", default='half')
    parser.add_argument("--low_reward_count", default='half')
    parser.add_argument("--mode", default='deterministic', type=str)
    parser.add_argument("--reset_radius", default=None, type=float)
    parser.add_argument("--filter", dest='filter', action='store_true')
    parser.add_argument("--filter_radius", default=0.2, type=float)
    parser.add_argument("--filter_pos", default=0, type=float)

    parser.set_defaults(new_exp=True)
    parser.set_defaults(verbose=True)
    parser.set_defaults(replay_buffer_visu=False)
    parser.set_defaults(policy_visu=True)
    parser.set_defaults(acceleration=False)
    parser.set_defaults(continuous=True)
    parser.set_defaults(render=True)
    parser.set_defaults(save=False)
    parser.set_defaults(filter=False)

    args = parser.parse_args()

    if args.save:
        path = learn_policy.setup_output_dir(args.output)
    else:
        path=args.output


    learn(algorithm=args.algorithm,
            seed=args.seed,
            dimensions=args.dimensions,
            eval_freq=args.eval_freq,
            exploration_timesteps=args.exploration_timesteps,
            exploration_mode=args.exploration_mode,
            learning_timesteps=args.learning_timesteps,
            buffer_size=args.buffer_size,
            new_exp=args.new_exp,
            expl_noise=args.expl_noise,
            batch_size=args.batch_size,
            discount=args.discount,
            actor_dim=(args.actor_hl1,args.actor_hl2),
            critic_dim=(args.critic_hl1,args.critic_hl2),
            learning_rate=args.learning_rate,
            tau=args.tau,
            policy_noise=args.policy_noise,
            noise_clip=args.noise_clip,
            policy_freq=args.policy_freq,
            verbose=args.verbose,
            replay_buffer_visu=args.replay_buffer_visu,
            policy_visu=args.policy_visu,
            acceleration=args.acceleration,
            continuous=args.continuous,
            render=args.render,
            save=args.save,
            output=path,
            high_reward_value=args.high_reward_value,
            low_reward_value=args.low_reward_value,
            high_reward_count=args.high_reward_count,
            low_reward_count=args.low_reward_count,
            mode=args.mode,
            reset_radius = args.reset_radius,
            filter=args.filter,
            filter_pos=args.filter_pos,
            filter_radius=args.filter_radius
            )

    if args.save:
        learn_policy.save_arguments(vars(args), path)

