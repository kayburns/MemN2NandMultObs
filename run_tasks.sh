#!/bin/bash
#
# To install the latest version of parallel, run
# (wget -O - pi.dk/3 || curl pi.dk/3/ || fetch -o - http://pi.dk/3) | bash

GPU_IDS=( 0 1 2 )
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
WORLD_SIZE_OPTS="0" # temporarily changed to noise
SEARCH_PROB_OPTS="1.00"
EXIT_PROB_OPTS="0.67"
INFORM_PROB_OPTS="0.50"
# TASK_ID_OPTS="21 22 23 24 25"
TASK_ID_OPTS="21"

SHA=$(git log --pretty=format:'%h' -n 1)
DATE=`date +%Y-%m-%d`

parallel -j ${#GPU_IDS[@]} \
'export CUDA_VISIBLE_DEVICES=$(({%} - 1)) && \
python main.py -te -ne 100 \
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
-d data/sally_anne/world_large_nex_100_{11} \
-o results/{12} \
-t {13}' \
::: $NONLIN_OPTS   `# 1` \
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
::: ${SHA}  `# 12` \
::: ${TASK_ID_OPTS}  `# 13` \
::: {1..$NUM_SIMS}  `# 14`
