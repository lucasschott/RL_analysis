#!/bin/sh

POLICY_NAME="DDPG"

START_TIMESTEP=1000

MAX_TIMESTEP=1500

BUFFER_SIZE=1000

EVAL_FREQ=50

MIN_DIMENSION=1

MAX_DIMENSION=10

ROOT_DIR="$(pwd)/"

RESULT_DIR="results/"

MODE="velocity"

run_training()
{
  OUTPUT_DIR="${ROOT_DIR}${RESULT_DIR}${POLICY_NAME}_n$1"

  COMMAND="python ../../learn_multidimensional.py\
    --policy_name=$POLICY_NAME\
    --start_timesteps=$START_TIMESTEP\
    --max_timesteps=$MAX_TIMESTEP\
    --buffer_size=$BUFFER_SIZE\
    --eval_freq=$EVAL_FREQ\
    --dimensions=$1\
    --${MODE}\
    --save\
    --no-render\
    --output=${OUTPUT_DIR}"

  eval ${COMMAND}
}


for i in $(seq $MIN_DIMENSION $MAX_DIMENSION)
  do
    echo "Training $i"
    run_training $i
  done

