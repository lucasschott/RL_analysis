#!/bin/sh

python ../../learn_multidimensional.py --policy_name=DDPG --exploration_timesteps=20000 --learning_timesteps=5000 --buffer_size=5000 --eval_freq=100 --seed=100 --no-new-exp --dimensions=2 --velocity --save --output=./results/
