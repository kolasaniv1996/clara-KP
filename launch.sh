#!/bin/bash

master_addr=$MASTER_ADDR
master_port=$MASTER_PORT
job_n=$WORLD_SIZE
#job_id=$RANK
# export RANK=$RANK  # Uncomment if you need to use RANK explicitly

# Optional: Print the variables for debugging
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "WORLD_SIZE: ${WORLD_SIZE}"
# echo "RANK: ${RANK}"

# Run the training script using torchrun
LOGLEVEL="INFO" torchrun --nproc_per_node=1 --nnodes=2:${job_n} --max-restarts=3 --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} --rdzv_backend=c10d camelyon.py --batch_size 8 1000 2

