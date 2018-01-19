#!/bin/bash

<<<<<<< HEAD
GPU_IDS=( 0 1 2 )
=======
GPU_IDS=( 0 1 2 3 )
>>>>>>> 06066df892d5b16e93052e7512be5485a57891f8
NUM_SIMS=10

ENCODING_TYPE_OPTS="position_encoding"
SHARE_TYPE_OPTS="layerwise"
NONLIN_OPTS="relu"
DIM_MEMORY_OPTS="10 20 50"
DIM_EMB_OPTS="10 20 50"
NUM_CACHES_OPTS="1 3"
INIT_STDDEV_OPTS=0.1
LEARNING_RATE_OPTS="0.01"
MAX_GRAD_NORM_OPTS=40
NUM_HOPS_OPTS="1 2 3"
WORLD_SIZE_OPTS="large small"
#SEARCH_PROB_OPTS="0.00 0.50 1.00"
SEARCH_PROB_OPTS="1.00"
EXIT_PROB_OPTS="0.00 0.50 1.00"
INFORM_PROB_OPTS="0.00"

SHA=$(git log --pretty=format:'%h' -n 1)
DATE=`date +%Y-%m-%d`

#'export CUDA_VISIBLE_DEVICES=$(({%} - 1)) &&

parallel -j ${#GPU_IDS[@]} \
'export CUDA_VISIBLE_DEVICES=$(({%} - 1)) && \
python main.py -te -ne 100 \
-t 1 -t 2 -t 3 -t 4 -t 5 -t 6 -t 7 -t 8 -t 9 -t 10 -t 11 -t 12 -t 13 -t 14 -t 15 -t 16 -t 17 -t 18 -t 19 -t 20 \
-nl {1} \
-et {2} \
-st {3} \
-de {4} \
-dm {5} \
-nc {6} \
-is {7} \
-lr {8} \
-gn {9} \
-nh {10} \
<<<<<<< HEAD
-d data/sally_anne/world_{11}_nex_1000_exitp_{12}_searchp_{13}_informp_{14} \
-o results/{15}' \
::: $NONLIN_OPTS   `# 1` \
=======
-d data/sally_anne/world_{11}_nex_1000_exitp_{13}_searchp_{12} \
-o results/{14} \
--joint \
-t {15}' \
::: $NONLIN_OPTS \
>>>>>>> 06066df892d5b16e93052e7512be5485a57891f8
::: $ENCODING_TYPE_OPTS  `# 2` \
::: $SHARE_TYPE_OPTS  `# 3` \
::: $DIM_EMB_OPTS  `# 4` \
::: $DIM_MEMORY_OPTS  `# 5` \
::: $NUM_CACHES_OPTS  `# 6` \
::: $INIT_STDDEV_OPTS  `# 7` \
::: $LEARNING_RATE_OPTS  `# 8` \
::: $MAX_GRAD_NORM_OPTS  `# 9` \
::: $NUM_HOPS_OPTS  `# 10` \
::: $WORLD_SIZE_OPTS  `# 11` \
<<<<<<< HEAD
::: $EXIT_PROB_OPTS  `# 12` \
::: $SEARCH_PROB_OPTS  `# 13` \
::: $INFORM_PROB_OPTS  `# 14` \
::: ${SHA}  `# 15` \
=======
::: $SEARCH_PROB_OPTS  `# 12` \
::: $EXIT_PROB_OPTS  `# 13` \
::: ${SHA}  `# 14` \
::: ${TASK_ID_OPTS}  `# 15` \
>>>>>>> 06066df892d5b16e93052e7512be5485a57891f8
::: {1..$NUM_SIMS}  `# 16`
