import torch.multiprocessing as mp
import os
import torch.distributed as dist
import torch
from torch import optim
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from datetime import datetime, timedelta
from Nets import nets, PreActResNets
from torch.nn.parallel import DistributedDataParallel as DDP
import itertools
import openslide
from utils import get_optimal_slide_level
from torchvision.transforms import ToTensor
import wandb
import random 
from torch.multiprocessing import Queue, Process
import socket
import argparse


"""
Proof of concept for low dataloading latency DDP training on our mrxs data with increased randomness.
Files are partitioned between workers and each worker only reads from their allocated files to a shared queue used for batch creation using a buffer. Random items from the buffer to create a new batch. The bottleneck proves to be the creation of a new batch from arbitrary idices. This is just one of many attempts: batching with Dataset, batching with numpy, permuting the cache. All proved inefficient for a yet to be discovered reason.
"""


def partition (list_in, n):
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    
    # initialize the process group
    dist.init_process_group("NCCL", rank=rank, world_size=world_size, timeout=timedelta(100))

def cleanup():
    dist.destroy_process_group()
    
def training_loop(rank, world_size, patch_queue, targets, batch_size, dataloader_workersm, verbose=False):
    try:
        setup(rank, world_size)
        loader = WSIDataLoader(targets=targets, patch_queue=patch_queue, batch_size=batch_size, randomization_factor=dataloader_workers)
        print(f"Running basic DDP example on rank {rank}.")
        # create model and move it to GPU with id rank
        model = PreActResNets.PreActResNet50_Ron().to(rank)
        ddp_model = DDP(model, device_ids=[rank])
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        for epoch in range(10):
            print(f"epoch {epoch} for {rank}")
            start_time = datetime.now()
            for batch_ndx, sample in enumerate(loader):
                print(f"batch {batch_ndx} for {rank}")
                pre_batch_time = datetime.now()
                if batch_ndx>0:
                    print(f"dataloading time for {rank} is {pre_batch_time - post_batch_time}")
                else:
                    print(f"loader creation time for {rank} is {pre_batch_time - start_time}")
                # Every data instance is an input + label pair
                inputs, labels = sample
                inputs = inputs
                labels = labels.to(rank)
                if verbose:
                    transfer_time = datetime.now()
                    print(f"memory to gpu for {rank} time is {transfer_time - pre_batch_time}")

                # Zero your gradients for every batch!
                optimizer.zero_grad()
                if verbose:
                    zero_grad_time = datetime.now()
                    print(f"zero_grad time for {rank} is {zero_grad_time - transfer_time}")

                # Make predictions for this batch
                outputs = ddp_model(inputs)[0][:, 0:(batch_size+1)]
                if verbose:
                    outputs_time = datetime.now()
                    print(f"outputs time for {rank} is {outputs_time - zero_grad_time}")

                # Compute the loss and its gradients
                loss = loss_fn(outputs, labels)
                barrier = dist.barrier()
                loss.backward()
                if verbose:
                    backward_time = datetime.now()
                    print(f"backward time for {rank} is {backward_time - outputs_time}")

                # Adjust learning weights
                optimizer.step()
                post_batch_time = datetime.now()
                if verbose:
                    print(f"optimizer time for {rank} is {post_batch_time - backward_time}")
            delta_time = datetime.now() - start_time
            print(f"basic DDP example on rank {rank} finished {len(loader)*world_size} batches in {delta_time}.\n{delta_time/(len(loader)*world_size)} time per batch.")
    except KeyboardInterrupt:
        raise KeyboardInterrupt

        
    cleanup()

def run_training(fn, world_size, patch_queue, targets, batch_size, dataloader_workers, verbose=False):
    try:
        mp.spawn(fn,
                 args=(world_size, patch_queue, targets, batch_size, dataloader_workers, verbose),
                 nprocs=world_size,
                 join=True)
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    
class WSIDataLoader:
    def __init__(self, targets, patch_queue, batch_size, randomization_factor):
        self.targets = targets
        self.batch_size = batch_size
        self.patch_queue = patch_queue
        queue_tensor = self.patch_queue.get()
        self.patch_size = queue_tensor.size()[1:]
        self.queue_item_size = queue_tensor.size()[0]
        self.patch_queue.put(queue_tensor)
        self.next_batches = np.empty((self.batch_size * randomization_factor, *self.patch_size), dtype=np.float32)
        self.next_batch = np.empty((self.batch_size, *self.patch_size), dtype=np.float32)
        self.patch_cache = np.empty((self.queue_item_size, *self.patch_size), dtype=np.float32)
        self.cache_end = 0
        self.batch_indices = np.arange(self.next_batches.shape[0])
        self.get_next_samples(init=True) 
        # self.next_batches is now filled starting batches and permuted
        self.batch_indices = np.random.choice(self.next_batches.shape[0], self.batch_size)
        
    def __iter__(self):
        self.n = 0
        return self
    
    def __len__(self):
        return int(np.ceil(float(len(self.targets)) / self.batch_size))
    
    def __next__(self):
        if self.n <= len(self):
            before_prep_time = datetime.now()
            self.get_next_samples() 
            # self.next_batches is now filled with an additional batch and permuted
            self.n += 1
            before_perm_time = datetime.now()
            print(f"time to get another batch in: {before_perm_time - before_prep_time}")
            self.batch_indices = np.random.choice(self.next_batches.shape[0], self.batch_size)
            after_perm_time = datetime.now()
            print(f"time to randomize batch of size {self.batch_size} out of {self.next_batches.shape[0]} items: {after_perm_time - before_perm_time}")
            self.next_batch = torch.from_numpy(self.next_batches[self.batch_indices], )
            print(f"time to turn to batch of size {self.batch_size} out of {self.next_batches.shape[0]} items: {datetime.now() - after_perm_time}")
            return self.next_batch, self.targets[:self.batch_size]
        else:
            raise StopIteration
            
    def get_next_samples(self, init=False):
        n_patches_from_cache = min(self.cache_end, self.batch_size)
        
        # Get leftover patches from cache
        if n_patches_from_cache:
            self.next_batches[:n_patches_from_cache] = self.patch_cache[:n_patches_from_cache]
            self.patch_cache = np.roll(self.patch_cache, n_patches_from_cache, axis=0)
            self.cache_end -= n_patches_from_cache
            
        # Complete next batch using queue items
        n_patches_leftover = 0
        n_patches_inserted = n_patches_from_cache
        n_patches_to_insert = self.next_batches.shape[0] if init else self.batch_size
        while n_patches_inserted < n_patches_to_insert:
            queue_patches = np.asarray(self.patch_queue.get(), dtype=np.float32)
            curr_queue_item_size = queue_patches.shape[0]
            batch_transfer_ending = min(n_patches_to_insert,
                                           n_patches_inserted + curr_queue_item_size)
            n_patches_leftover = n_patches_inserted + curr_queue_item_size - n_patches_to_insert
            self.next_batches[self.batch_indices[n_patches_inserted:batch_transfer_ending]] =\
                queue_patches[:batch_transfer_ending - n_patches_inserted]
            n_patches_inserted = batch_transfer_ending
            
        # Put leftover patches in cache
        if n_patches_leftover:
            self.patch_cache[:n_patches_leftover] = queue_patches[-n_patches_leftover:]
            self.cache_end += n_patches_leftover
        

            
def queue_deamon(queue, file_list, best_slide_level, adjusted_tile_size):
    """Fetches files for queue"""
    slide_handles = []
    # files are preloaded since this takes the majority of time when handling files.
    for file in tqdm(file_list):
        slide_path = file
        slide_handle = openslide.open_slide(slide_path)
        slide_handles.append(slide_handle)
        
    daemon_tensor = torch.zeros(len(file_list), 3, adjusted_tile_size, adjusted_tile_size)
    # Below is an arbitrary choice of file & region to read from. Can be replaced by actual patch choice func.
    # Also smaller queue item sizes need to be tested. Size 1 seems to be hellishly inappropriate, but other sizes might be better. Smaller is better for randomness.
    while True:
        indecies = np.random.permutation(len(file_list))
        for progress_idx, file_idx in enumerate(indecies):
            slide = slide_handles[file_idx]
            im = slide.read_region((40000,40000),
                                 best_slide_level,
                                 (adjusted_tile_size, adjusted_tile_size)).convert('RGB')
            daemon_tensor[progress_idx] = ToTensor()(im)
        queue.put(daemon_tensor)
                
            
            
def init_queue(file_list, best_slide_level, adjusted_tile_size, num_workers):
    """Initializes queue and workers responsible for filling it."""
    file_lists = partition(file_list, num_workers)
    # Files are split between different workers simce having the same file open in two different processes appears to be causing stalling and deadlocks. Current split is arbitrary, and splitting based on balancing patches between workers is probably preferable.
    queue = Queue()
    # When is doubt about preformance consider changing to SimpleQueue
    workers = []
    for idx in range(num_workers):
        workers.append(Process(target=queue_deamon, args=(queue, file_lists[idx], best_slide_level, adjusted_tile_size), daemon=True))
        workers[-1].start()
    return queue, workers
        
    
parser = argparse.ArgumentParser(description='Worker Based Batching DDP POV')
parser.add_argument('--verbose', action='store_true', help='whether to print timing for dataloading')
args = parser.parse_args()
    
if __name__ == '__main__':
    mp.set_start_method('spawn')
    # Important for linux. Prevents deadlocks stemming from resource sharing.
    config = {"cpus":64,
              "gpus":torch.cuda.device_count(),
              "batch_size": 124
             }
    batch_size = config["batch_size"]
    cpus = config["cpus"]
    gpus = config["gpus"]
    print(config)
    data_dir = "/data" if socket.gethostname() == "gipdeep10" else "/mnt/gipmed_new/Data"
    image_dir = join(data_dir, "Breast/Haemek/Batch_1/HAEMEK1")
    onlyfiles = [join(image_dir, f) for f in listdir(image_dir) if isfile(join(image_dir, f)) and f.endswith('mrxs')]
    tgts = torch.ones(len(onlyfiles)*2, dtype=torch.float32).view(len(onlyfiles), 2) 
    slide = openslide.open_slide(onlyfiles[0])
    best_slide_level, adjusted_tile_size, level_0_tile_size = get_optimal_slide_level(slide, 40, 10, 256)
    queue, workers = init_queue(onlyfiles, best_slide_level, adjusted_tile_size, cpus)
    # Queue gets filled by the workers and is used as input stream for training processes initialized below.
    try:
        run_training(training_loop, gpus, queue, tgts, batch_size, cpus // gpus, args.verbose)
    except KeyboardInterrupt:
        print(workers)
        for worker in workers:
            worker.join()