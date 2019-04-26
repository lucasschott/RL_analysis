#!/bin/bash

## Performance en fonction du nombre de dimensions avec moitié high et moitié low reward

PARALLEL_MAX=2

MEAN_BATCH_SIZE=8

POLICY_NAME="DDPG"

EXPLORATION_TIMESTEPS=32000

LEARNING_TIMESTEPS_1=15000

LEARNING_TIMESTEPS_2=3000

BUFFER_SIZE=32000

EVAL_FREQ=300

EXPLORATION_MODE="uniform"

RESET_RADIUS=0

ROOT_DIR="$(pwd)/"

RESULT_DIR="results_convergence_reset_indept/"

HIGH_REWARD_COUNT="half"

LOW_REWARD_COUNT="half"

SPEED_LIMIT="independent"

TITLE="dimensions"

run_training_1()
{
  OUTPUT_DIR="${ROOT_DIR}${RESULT_DIR}${POLICY_NAME}_n$1_$2"

  COMMAND="python ../../learn_hypercube.py\
    --algorithm=$POLICY_NAME\
    --exploration_timesteps=$EXPLORATION_TIMESTEPS\
    --learning_timesteps=$LEARNING_TIMESTEPS_1\
    --buffer_size=$BUFFER_SIZE\
    --eval_freq=$EVAL_FREQ\
    --dimensions=$1\
    --save\
    --no_policy_visu\
    --no_render\
    --reset_radius=$RESET_RADIUS\
    --exploration_mode=${EXPLORATION_MODE}\
    --high_reward_count=$HIGH_REWARD_COUNT\
    --low_reward_count=$LOW_REWARD_COUNT\
    --speed_limit_mode=$SPEED_LIMIT\
    --output=${OUTPUT_DIR}\
    --reset_radius=$RESET_RADIUS"

  eval ${COMMAND}
}

run_training_2()
{
  OUTPUT_DIR="${ROOT_DIR}${RESULT_DIR}${POLICY_NAME}_n$1_$2"

  COMMAND="python ../../learn_hypercube.py\
    --algorithm=$POLICY_NAME\
    --exploration_timesteps=$EXPLORATION_TIMESTEPS\
    --learning_timesteps=$LEARNING_TIMESTEPS_2\
    --buffer_size=$BUFFER_SIZE\
    --eval_freq=$EVAL_FREQ\
    --dimensions=$1\
    --save\
    --no_policy_visu\
    --no_render\
    --exploration_mode=${EXPLORATION_MODE}\
    --high_reward_count=$HIGH_REWARD_COUNT\
    --low_reward_count=$LOW_REWARD_COUNT\
    --speed_limit_mode=$SPEED_LIMIT\
    --output=${OUTPUT_DIR}\
    --reset_radius=$RESET_RADIUS"

  eval ${COMMAND}
}


PARALLEL=0
PIDS=()

for i in 1 2 4 8 16 32 64 128
do
    for j in $(seq 0 $(($MEAN_BATCH_SIZE-1)))
    do
	echo "Training $i $j"
	if [ $i -le 4 ]
	then
		run_training_1 $i $j &
	else
		run_training_2 $i $j &
	fi
	PIDS[$PARALLEL]=$!

        PARALLEL=$(($PARALLEL+1))
        if [ $PARALLEL -ge $PARALLEL_MAX ]
        then
            PARALLEL=0
	    wait ${PIDS[@]}
	    PIDS=()
        fi
    done
done
wait ${PIDS[@]}

COMMAND4="python ../plot_average_convergence.py\
    --directory=$RESULT_DIR\
    --batch_size=$MEAN_BATCH_SIZE\
    --eval_freq=$EVAL_FREQ\
    --epsilon=0.02\
    --title='$TITLE'"

eval ${COMMAND4}
