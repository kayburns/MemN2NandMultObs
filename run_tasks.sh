#!/bin/bash

GPU_IDS=( 0 1 2 )
NUM_SIMS=10

ENCODING_TYPE_OPTS="position_encoding"
NUM_HOPS_OPTS="1 2 3 4 5"
NONLIN_OPTS="relu"
DIM_MEMORY_OPTS="10 20 50 100"
DIM_EMB_OPTS="10 20 50 100"
NUM_CACHES_OPTS=1
INIT_STDDEV_OPTS=0.1
LEARNING_RATE_OPTS="0.1 0.01 0.001"
MAX_GRAD_NORM_OPTS=40
WORLD_SIZE_OPTS="large small tiny"
SEARCH_PROB_OPTS="0.00 0.50 1.00"
EXIT_PROB_OPTS="0.00 0.50 1.00"
TASK_ID_OPTS="1 2 3 4 5"

parallel -j ${#GPU_IDS[@]} 'export CUDA_VISIBLE_DEVICES=$(({%} - 1)) &&\
python main.py -te -ne 100 \
-nh {1} \
-nl {2} \
-et {3} \
-dm {4} \
-de {5} \
-nc {6} \
-is {7} \
-lr {8} \
-gn {9} \
-t {10} \
-d data/sally_anne/world_{11}_nex_1000_exitp_{12}_searchp_{13}' \
::: $NUM_HOPS_OPTS \
::: $NONLIN_OPTS \
::: $ENCODING_TYPE_OPTS \
::: $DIM_MEMORY_OPTS \
::: $DIM_EMB_OPTS \
::: $NUM_CACHES_OPTS \
::: $INIT_STDDEV_OPTS \
::: $LEARNING_RATE_OPTS \
::: $MAX_GRAD_NORM_OPTS \
::: $TASK_ID_OPTS \
::: $WORLD_SIZE_OPTS \
::: $SEARCH_PROB_OPTS \
::: $EXIT_PROB_OPTS \
::: {1..$NUM_SIMS}
