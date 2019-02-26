#!/bin/sh

## performance en fonction du nombre de dimensions avec une unique high et low reward

PARALLEL_MAX=1

AVERAGE_NB=1

POLICY_NAME="DDPG"

EXPLORATION_TIMESTEPS=1000

LEARN_TIMESTEPS=1000

BUFFER_SIZE=1000

EVAL_FREQ=500

MIN_DIMENSION=1

MAX_DIMENSION=10

ROOT_DIR="$(pwd)/"

RESULT_DIR="results/"

MODE="velocity"

TITLE="Performance d apprentissage en fonction du nombre de dimensions (une unique high et low reward)"

X_LABEL="Nombre de dimensions"

Y_LABEL="Reward moyen par step"

HIGH_REWARD_COUNT="one"

LOW_REWARD_COUNT="one"

run_training()
{
  OUTPUT_DIR="${ROOT_DIR}${RESULT_DIR}${POLICY_NAME}_n$1"

  COMMAND="python ../../learn_multidimensional.py\
    --policy_name=$POLICY_NAME\
    --exploration_timesteps=$EXPLORATION_TIMESTEPS\
    --learn_timesteps=$LEARN_TIMESTEPS\
    --buffer_size=$BUFFER_SIZE\
    --eval_freq=$EVAL_FREQ\
    --dimensions=$1\
    --${MODE}\
    --save\
    --no-render\
    --high_reward_count=$HIGH_REWARD_COUNT\
    --low_reward_count=$LOW_REWARD_COUNT\
    --output=${OUTPUT_DIR}"

  eval ${COMMAND}
}


for i in $(seq $MIN_DIMENSION $MAX_DIMENSION)
do
    for j in $(seq 0 $(($AVERAGE_NB-1)))
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
    --average_nb=$AVERAGE_NB\
    --title='$TITLE'\
    --x_label='$X_LABEL'\
    --y_label='$Y_LABEL'"

eval ${COMMAND2}
