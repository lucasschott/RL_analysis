#!/bin/sh

## Performance en fonction du nombre de dimensions avec moitié high et moitié low reward

PARALLEL_MAX=8

MEAN_BATCH_SIZE=4

POLICY_NAME="DDPG"

EXPLORATION_TIMESTEPS=5000

LEARNING_TIMESTEPS=20000

BUFFER_SIZE=5000

EVAL_FREQ=1000

DIMENSION=2

LEARNING_RATE=0.0001

MIN_TAU=0

TAU_INCREASE_STEP=0.2

MAX_TAU=1

ROOT_DIR="$(pwd)/"

RESULT_DIR="results/"

MODE="velocity"

TITLE=""

X_LABEL="tau"

Y_LABEL="reward/step"

HIGH_REWARD_COUNT="one"

LOW_REWARD_COUNT="one"

run_training()
{
  OUTPUT_DIR="${ROOT_DIR}${RESULT_DIR}${POLICY_NAME}_n$1_$2"

  COMMAND="python ../../learn_multidimensional.py\
    --policy_name=$POLICY_NAME\
    --exploration_timesteps=$EXPLORATION_TIMESTEPS\
    --learning_timesteps=$LEARNING_TIMESTEPS\
    --buffer_size=$BUFFER_SIZE\
    --eval_freq=$EVAL_FREQ\
    --tau=$1\
    --learning_rate=$LEARNING_RATE\
    --dimensions=$DIMENSION\
    --${MODE}\
    --save\
    --no-render\
    --high_reward_count=$HIGH_REWARD_COUNT\
    --low_reward_count=$LOW_REWARD_COUNT\
    --output=${OUTPUT_DIR}"

  eval ${COMMAND}
}


PARALLEL=0

for i in $(seq $MIN_TAU $TAU_INCREASE_STEP $MAX_TAU)
do
    for j in $(seq 0 $(($MEAN_BATCH_SIZE-1)))
    do
        PARALLEL=$(($PARALLEL+1))
        if [ $PARALLEL -ge $PARALLEL_MAX ]
        then
            echo "Training $i $j"
            run_training $i $j
            PARALLEL=0
        else
            echo "Training $i $j"
            run_training $i $j &
        fi
    done
done


COMMAND2="python ../plot_evaluations.py\
    --directory=$RESULT_DIR\
    --policy_name=$POLICY_NAME\
    --batch_size=$MEAN_BATCH_SIZE\
    --title='$TITLE'\
    --x_label='$X_LABEL'\
    --y_label='$Y_LABEL'"

eval ${COMMAND2}
