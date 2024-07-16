# clara-camelyon16

# Distributed Training with Camelyon16 Dataset

This project sets up distributed training for a model using the Camelyon16 dataset. It leverages PyTorch's distributed training capabilities and the dataset available on Hugging Face.

## Prerequisites

- Python 3.6+
- PyTorch
- Hugging Face `datasets` library

## Setup

### Install Dependencies

1. Install PyTorch following the instructions on [PyTorch's website](https://pytorch.org/get-started/locally/).
2. Install the Hugging Face `datasets` library:

```bash
pip install datasets

Usage
Environment Variables
Before running the training script, ensure the following environment variables are set:

MASTER_ADDR: The address of the master node.
MASTER_PORT: The port on which the master node will communicate.
WORLD_SIZE: The total number of nodes participating in the job.

camelyon.py script sets up distributed training using PyTorch's torchrun. The dataset is loaded from Hugging Face.




RUNAI CLI Command :-


runai submit-dist pytorch --name clara-clara-camelyon16 --workers=3 --max-replicas 4 --min-replicas 2 -g 1 -i docker.io/vivekkolasani1996/clara-4:v1


docker.io/vivekkolasani1996/clara-4:v1 is the image build from the dockerfile
