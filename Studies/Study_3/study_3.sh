#!/bin/sh

python ../../learn_multidimensional.py --policy_name=DDPG --exploration_timesteps=10000 --learning_timesteps=50000 --buffer_size=10000 --eval_freq=1000 --seed=100 --no-new-exp --dimensions=2 --velocity --save --output=./results/ --filter=circle
