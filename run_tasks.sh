#!/bin/bash

GPU_IDS=( 0 1 )
NUM_SIMS=10

ENCODING_TYPE_OPTS="position_encoding"
SHARE_TYPE_OPTS="adjacent layerwise"
NONLIN_OPTS="relu"
DIM_MEMORY_OPTS="5 10 20 50 100"
DIM_EMB_OPTS="5 10 20 50 100 200"
NUM_CACHES_OPTS=1
INIT_STDDEV_OPTS=0.1
LEARNING_RATE_OPTS="0.01"
MAX_GRAD_NORM_OPTS=40
NUM_HOPS_OPTS="1 2 3 4 5"
WORLD_SIZE_OPTS="large small tiny"
SEARCH_PROB_OPTS="0.00 0.50 1.00"
EXIT_PROB_OPTS="0.00 0.25 0.50 0.75 1.00"
TASK_ID_OPTS="21 22 23 24 25"

SHA=$(git log --pretty=format:'%h' -n 1)
DATE=`date +%Y-%m-%d`

parallel --joblog $DATE_$SHA -j ${#GPU_IDS[@]} \
'export CUDA_VISIBLE_DEVICES=$(({%} - 1)) &&
python main.py -te -ne 100 \
-t 21 -t 22 -t 23 -t 24 -t 25 \
-nl {1} \
-et {2} \
-st {3} \
-dm {4} \
-de {5} \
-nc {6} \
-is {7} \
-lr {8} \
-gn {9} \
-nh {10} \
-d data/sally_anne/world_{11}_nex_1000_exitp_{12}_searchp_{13} \
-o results/{14}' \
::: $NONLIN_OPTS \
::: $ENCODING_TYPE_OPTS \
::: $SHARE_TYPE_OPTS \
::: $DIM_MEMORY_OPTS \
::: $DIM_EMB_OPTS \
::: $NUM_CACHES_OPTS \
::: $INIT_STDDEV_OPTS \
::: $LEARNING_RATE_OPTS \
::: $MAX_GRAD_NORM_OPTS \
::: $NUM_HOPS_OPTS \
::: $WORLD_SIZE_OPTS \
::: $SEARCH_PROB_OPTS \
::: $EXIT_PROB_OPTS \
::: ${SHA} \
::: {1..$NUM_SIMS} 
