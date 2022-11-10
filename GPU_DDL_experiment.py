import torch.multiprocessing as mp
import os
import torch.distributed as dist
import torch
from torch.utils import data
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from datetime import datetime
from Nets import nets, PreActResNets, resnet_v2
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '6007'
    
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    
def demo_basic(rank, world_size, dataset):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)
        
    # create model and move it to GPU with id rank
    model = PreActResNets.PreActResNet50_Ron().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loader = data.DataLoader(dataset, batch_size=129, pin_memory=True, num_workers=8)
    
    start_time = datetime.now()
    
    for batch_ndx, sample in enumerate(loader):
        # Every data instance is an input + label pair
        inputs, labels = sample
        inputs = inputs.to(rank)
        labels = labels.to(rank)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = ddp_model(inputs)[0][:, 0:2]

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()
        
    delta_time = datetime.now() - start_time
    
    print(f"basic DDP example on rank {rank} finished {len(loader)*world_size} batches in {delta_time}.\n{delta_time/(len(loader)*world_size)} time per batch.")
    cleanup()

def run_demo(demo_fn, world_size, dataset):
    mp.spawn(demo_fn,
             args=(world_size,dataset),
             nprocs=world_size,
             join=True)
    
if __name__ == '__main__':
    TMA_array_path = '/data/Breast/TMA/bliss_data/17-002/Ki67[SP6]-Thermo'
    onlyfiles = [f for f in listdir(TMA_array_path) if isfile(join(TMA_array_path, f)) and f.endswith('jpg')]
    
    inps = []
    img_size = 256
    for im_name in tqdm(onlyfiles):
        im = cv2.imread(join(TMA_array_path, im_name))
        (height,width)=im.shape[:2]
        im = im[int(height/2.0-img_size/2.0):int(height/2.0+img_size/2.0), 
                int(width/2.0-img_size/2.0):int(width/2.0+img_size/2.0),
                :]
        inps.append(np.transpose(im, axes=(2, 0, 1)))
    inps = torch.FloatTensor(inps)
    inps = inps.repeat(10, 1, 1, 1)
    print(inps.shape)
    tgts = torch.ones(len(inps)*2, dtype=torch.float32).view(len(inps), 2)
    dataset = data.TensorDataset(inps, tgts)
    run_demo(demo_basic, world_size=4, dataset=dataset)