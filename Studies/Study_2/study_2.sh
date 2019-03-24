#!/bin/sh

## Performance en fonction de la taille du buffer

PARALLEL_MAX=8

MEAN_BATCH_SIZE=4

POLICY_NAME="DDPG"

EXPLORATION_TIMESTEPS=65536

LEARNING_TIMESTEPS=20000

EVAL_FREQ=100

TAU=0.5

DIMENSION=2

ROOT_DIR="$(pwd)/"

RESULT_DIR="results/"

MODE="velocity"

TITLE=""

X_LABEL="replay buffer size"

Y_LABEL="reward/step"


run_training()
{
  OUTPUT_DIR="${ROOT_DIR}${RESULT_DIR}${POLICY_NAME}_n$1_$2"

  COMMAND="python ../../learn_multidimensional.py\
    --policy_name=$POLICY_NAME\
    --exploration_timesteps=$EXPLORATION_TIMESTEPS\
    --learning_timesteps=$LEARNING_TIMESTEPS\
    --buffer_size=$1\
    --eval_freq=$EVAL_FREQ\
    --tau=$TAU\
    --dimensions=$DIMENSION\
    --${MODE}\
    --save\
    --no-render\
    --no-new-exp\
    --output=${OUTPUT_DIR}"

  eval ${COMMAND}
}


PARALLEL=0

for i in 64 128 256 512 1024 2048 4096 8192 16384 32768 65536
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
    --y_label='$Y_LABEL'\
    --log_scale"

eval ${COMMAND2}
