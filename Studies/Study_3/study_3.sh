#!/bin/sh

python ../../learn_multidimensional.py --policy_name=DDPG --exploration_timesteps=10000 --learning_timesteps=20000 --buffer_size=10000 --eval_freq=2000 --seed=100 --no-new-exp --dimensions=2 --velocity --save --output=./results/filter --filter=circle --reset_radius=0.2 &

python ../../learn_multidimensional.py --policy_name=DDPG --exploration_timesteps=10000 --learning_timesteps=20000 --buffer_size=10000 --eval_freq=2000 --seed=100 --no-new-exp --dimensions=2 --velocity --save --output=./results/no_filter --reset_radius=0.2
