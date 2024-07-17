#!/bin/bash

# Set the environment variables
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export WORLD_SIZE=$WORLD_SIZE
# export RANK=$RANK  # Uncomment if you need to use RANK explicitly

# Optional: Print the variables for debugging
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "WORLD_SIZE: ${WORLD_SIZE}"
# echo "RANK: ${RANK}"

# Run the training script using torchrun
torchrun --nproc_per_node=1 --nnodes=${WORLD_SIZE} --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} --rdzv_backend=c10d camelyon.py --batch_size 8 30 2
