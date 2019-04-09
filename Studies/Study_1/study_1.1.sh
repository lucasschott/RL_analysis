#!/bin/bash

## Performance en fonction de la taille du buffer

PARALLEL_MAX=2

MEAN_BATCH_SIZE=8

POLICY_NAME="DDPG"

LEARNING_TIMESTEPS=40000

EVAL_FREQ=2000

DIMENSION=2

EXPLORATION_MODE="uniform"

RESET_RADIUS=0.1

ROOT_DIR="$(pwd)/"

RESULT_DIR="results/"

TITLE="replay buffer size"

X_LABEL="replay buffer size"

Y_LABEL="reward/step"

run_training()
{
  OUTPUT_DIR="${ROOT_DIR}${RESULT_DIR}${POLICY_NAME}_n$1_$2"

  COMMAND="python ../../learn_hypercube.py\
    --policy_name=$POLICY_NAME\
    --exploration_timesteps=$1\
    --learning_timesteps=$LEARNING_TIMESTEPS\
    --buffer_size=$1\
    --eval_freq=$EVAL_FREQ\
    --dimensions=$DIMENSION\
    --save\
    --no-policy_visu\
    --no-render\
    --exploration_mode=${EXPLORATION_MODE}\
    --output=${OUTPUT_DIR}\
    --reset_radius=$RESET_RADIUS\
    --no-new-exp"

  eval ${COMMAND}
}


PARALLEL=0
PIDS=()

for i in 16 64 256 1024 4096 16384 65536
do
    for j in $(seq 0 $(($MEAN_BATCH_SIZE-1)))
    do
	echo "Training $i $j"
	run_training $i $j &
	PIDS[$j]=$!

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

COMMAND2="python ../plot_evaluations.py\
    --directory=$RESULT_DIR\
    --batch_size=$MEAN_BATCH_SIZE\
    --title='$TITLE'\
    --x_label='$X_LABEL'\
    --y_label='$Y_LABEL'\
    --log_scale"

eval ${COMMAND2}

COMMAND3="python ../plot_average_q.py\
    --directory=$RESULT_DIR\
    --batch_size=$MEAN_BATCH_SIZE\
    --learning_timesteps=$LEARNING_TIMESTEPS\
    --eval_freq=$EVAL_FREQ\
    --title='$TITLE'"

eval ${COMMAND3}

COMMAND4="python ../plot_average_pi.py\
    --directory=$RESULT_DIR\
    --batch_size=$MEAN_BATCH_SIZE\
    --learning_timesteps=$LEARNING_TIMESTEPS\
    --eval_freq=$EVAL_FREQ\
    --title='$TITLE'"

eval ${COMMAND4}

COMMAND5="python ../plot_average_learning_curve.py\
    --directory=$RESULT_DIR\
    --batch_size=$MEAN_BATCH_SIZE\
    --eval_freq=$EVAL_FREQ\
    --title='$TITLE'"

eval ${COMMAND5}
