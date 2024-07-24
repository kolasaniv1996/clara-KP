This repository contains a script for training a UNet model for brain MRI segmentation using a distributed training setup with PyTorch. The script utilizes the Clara image for loading and processing the data.

## Requirements

- Python 3.7+
- PyTorch 1.7.1+
- Torchvision 0.8.2+
- PIL (Pillow)
- numpy

## Setup

1. **Clone the Repository**:

    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Install Dependencies**:

    Install the required Python packages:

    ```sh
    pip install torch torchvision pillow numpy
    ```

3. **Download the Model and Dataset**:

    The script will automatically download the pre-trained model and the sample brain MRI image dataset when you run it.

## Usage

### Running the Training Script

To start the training process, simply run the script:

```sh
python train.py
