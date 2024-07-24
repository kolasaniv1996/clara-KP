import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def download_model():
    # Download and save the model to a specific path
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
        in_channels=3, out_channels=1, init_features=32, pretrained=True)
    model_path = "/home/local/data/unet_model.pth"
    torch.save(model.state_dict(), model_path)
    return model_path

def load_image():
    url, filename = ("https://github.com/mateuszbuda/brain-segmentation-pytorch/raw/master/assets/brain-mri-lgg.png", "/home/local/data/brain-mri-lgg.png")
    if not os.path.exists(filename):
        import urllib.request
        urllib.request.urlretrieve(url, filename)
    return filename

def preprocess_image(filename):
    input_image = Image.open(filename).convert("RGB")
    input_image = input_image.resize((256, 256))
    input_array = np.array(input_image)

    # Normalize image
    mean, std = input_array.mean(), input_array.std()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean/255.0]*3, std=[std/255.0]*3)
    ])
    input_tensor = transform(input_image).unsqueeze(0)
    return input_tensor

def create_target_mask(input_tensor):
    # Create a dummy target mask for the purpose of this example
    _, _, h, w = input_tensor.size()
    target_mask = torch.zeros(1, 1, h, w)
    # Creating a fake tumor region
    target_mask[:, :, 100:150, 100:150] = 1.0
    return target_mask

def train(rank, world_size, model_path):
    print(f"Running on rank {rank}", flush=True)
    setup(rank, world_size)
    
    # Load your model from the local file
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
        in_channels=3, out_channels=1, init_features=32, pretrained=False)
    model.load_state_dict(torch.load(model_path))

    # Wrap model with DDP without specifying device_ids
    model = DDP(model)

    # Load data
    filename = load_image()
    input_tensor = preprocess_image(filename)
    target_mask = create_target_mask(input_tensor)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Training loop
    for epoch in range(10):
        model.train()
        optimizer.zero_grad()
        output = model(input_tensor)
        loss = criterion(output, target_mask)
        loss.backward()
        optimizer.step()
        if rank == 0:
            print(f"Epoch [{epoch+1}/10], Loss: {loss.item()}", flush=True)

    cleanup()

def main():
    world_size = 2  # Number of processes (use more if you have more CPU cores)
    model_path = download_model()  # Ensure the model is downloaded once before starting multiprocessing
    mp.spawn(train, args=(world_size, model_path), nprocs=world_size, join=True)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)  # Ensures spawn method is used
    main()
