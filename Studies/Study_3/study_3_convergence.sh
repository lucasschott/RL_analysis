#!/bin/bash

## Performance en fonction du nombre de dimensions avec moitié high et moitié low reward

PARALLEL_MAX=2

MEAN_BATCH_SIZE=8

POLICY_NAME="DDPG"

EXPLORATION_TIMESTEPS=32000

LEARNING_TIMESTEPS_1=24000

LEARNING_TIMESTEPS_2=3000

BUFFER_SIZE=32000

EVAL_FREQ_1=1200

EVAL_FREQ_2=100

EXPLORATION_MODE="uniform"

RESET_RADIUS=0.1

ROOT_DIR="$(pwd)/"

RESULT_DIR="results_convergence/"

HIGH_REWARD_COUNT="half"

LOW_REWARD_COUNT="half"

TITLE="dimensions"

run_training_1()
{
  OUTPUT_DIR="${ROOT_DIR}${RESULT_DIR}${POLICY_NAME}_n$1_$2"

  COMMAND="python ../../learn_multidimensional.py\
    --policy_name=$POLICY_NAME\
    --exploration_timesteps=$EXPLORATION_TIMESTEPS\
    --learning_timesteps=$LEARNING_TIMESTEPS_1\
    --buffer_size=$BUFFER_SIZE\
    --eval_freq=$EVAL_FREQ_1\
    --dimensions=$1\
    --save\
    --no-policy_visu\
    --no-render\
    --exploration_mode=${EXPLORATION_MODE}\
    --high_reward_count=$HIGH_REWARD_COUNT\
    --low_reward_count=$LOW_REWARD_COUNT\
    --output=${OUTPUT_DIR}\
    --reset_radius=$RESET_RADIUS"

  eval ${COMMAND}
}

run_training_2()
{
  OUTPUT_DIR="${ROOT_DIR}${RESULT_DIR}${POLICY_NAME}_n$1_$2"

  COMMAND="python ../../learn_multidimensional.py\
    --policy_name=$POLICY_NAME\
    --exploration_timesteps=$EXPLORATION_TIMESTEPS\
    --learning_timesteps=$LEARNING_TIMESTEPS_2\
    --buffer_size=$BUFFER_SIZE\
    --eval_freq=$EVAL_FREQ_2\
    --dimensions=$1\
    --save\
    --no-policy_visu\
    --no-render\
    --exploration_mode=${EXPLORATION_MODE}\
    --high_reward_count=$HIGH_REWARD_COUNT\
    --low_reward_count=$LOW_REWARD_COUNT\
    --output=${OUTPUT_DIR}\
    --reset_radius=$RESET_RADIUS"

  eval ${COMMAND}
}


PARALLEL=0
PIDS=()

for i in 2 4 8 16 32 64 128 256
do
    for j in $(seq 0 $(($MEAN_BATCH_SIZE-1)))
    do
	echo "Training $i $j"
	if [ $i -lt 16 ]
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
    --eval_freq_1=$EVAL_FREQ_1\
    --eval_freq_2=$EVAL_FREQ_2\
    --epsilon=0.02\
    --log_scale\
    --title='$TITLE'"

eval ${COMMAND4}
