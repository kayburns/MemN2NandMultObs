#!/bin/bash

ENCODING_TYPES="position_encoding"
NUMS_HOPS="1 2 3 4 5"
NONLINS="relu"
DIMS_MEMORY="20 50 100"
DIMS_EMB="10 20 50 100"
NUMS_CACHES=1
INITS_STDDEV=0.1
LEARNING_RATES="0.1 0.01 0.001"
MAX_GRAD_NORMS=40
WORLD_SIZES="large"
SEARCH_PROBS="0.00 0.50 1.00"
EXIT_PROBS="0.00 0.50 1.00"

for WORLD_SIZE in $WORLD_SIZES; do
for NONLIN in $NONLINS; do
for NUM_HOPS in $NUMS_HOPS; do
for ENCODING_TYPE in $ENCODING_TYPES; do
for DIM_MEMORY in $DIMS_MEMORY; do
for DIM_EMB in $DIMS_EMB; do
for NUM_CACHES in $NUMS_CACHES; do
for INIT_STDDEV in $INITS_STDDEV; do
for LEARNING_RATE in $LEARNING_RATES; do
for MAX_GRAD_NORM in $MAX_GRAD_NORMS; do
for EXIT_PROB in $EXIT_PROBS; do
for SEARCH_PROB in $SEARCH_PROBS; do
for TASK_ID in 2 3 4 5; do

DATA_PATH="data/sally_anne/world_${WORLD_SIZE}_nex_1000_exitp_${EXIT_PROB}_searchp_${SEARCH_PROB}"

python main.py \
-nh $NUM_HOPS \
-nl $NONLIN \
-et $ENCODING_TYPE \
-dm $DIM_MEMORY \
-de $DIM_EMB \
-nc $NUM_CACHES \
-is $INIT_STDDEV \
-lr $LEARNING_RATE \
-gn $MAX_GRAD_NORM \
-te \
-ne 100 \
-t $TASK_ID \
-d $DATA_PATH

done
done
done
done
done
done
done
done
done
done
done
done
done
