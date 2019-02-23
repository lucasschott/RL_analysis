#!/bin/sh

## Performance en fonction de la taille du buffer

POLICY_NAME="DDPG"

EXPLORATION_TIMESTEPS=4000

LEARN_TIMESTEPS=1000

MIN_BUFFER=200

BUFFER_INCREASE_STEP=200

MAX_BUFFER=2000

EVAL_FREQ=500

DIMENSION=2

ROOT_DIR="$(pwd)/"

RESULT_DIR="results/"

MODE="velocity"

TITLE="Performance d apprentissage en fonction de la taille du replay buffer"

X_LABEL="Taille du replay buffer"

Y_LABEL="Reward moyen par step"


run_training()
{
  OUTPUT_DIR="${ROOT_DIR}${RESULT_DIR}${POLICY_NAME}_n$1"

  COMMAND="python ../../learn_multidimensional.py\
    --policy_name=$POLICY_NAME\
    --exploration_timesteps=$EXPLORATION_TIMESTEPS\
    --learn_timesteps=$LEARN_TIMESTEPS\
    --buffer_size=$1\
    --eval_freq=$EVAL_FREQ\
    --dimensions=$DIMENSION\
    --${MODE}\
    --save\
    --no-render\
    --no-new-exp\
    --output=${OUTPUT_DIR}"

  eval ${COMMAND}
}


for i in $(seq $MIN_BUFFER $BUFFER_INCREASE_STEP $MAX_BUFFER)
do
    echo "Training $i"
    run_training $i
done


COMMAND2="python ../plot_evaluations.py\
    --directory=$RESULT_DIR\
    --policy_name=$POLICY_NAME\
    --title='$TITLE'\
    --x_label='$X_LABEL'\
    --y_label='$Y_LABEL'"

eval ${COMMAND2}
