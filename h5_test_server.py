from pathlib import Path
from datetime import datetime
import time
import numpy as np

# gipmed
# from nn.experiments import Experiment, ExperimentArgumentsParser
from wsi_core import constants
from datasets.datasets import PatchSupervisedDataset
from models import preact_resnet
from wsi_core.wsi import BioMarker

# torch
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from torchvision import transforms
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

# matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines
# plt.rcParams["savefig.bbox"] = 'tight'

# pandas
import pandas

if __name__ == '__main__':

    def my_collate(batch):
        patches = [item[0] for item in batch]
        patches = torch.from_numpy(np.stack(patches))
        labels = torch.as_tensor([item[1] for item in batch])
        return patches, labels

    wsi_dataset = PatchSupervisedDataset()

    wsi_dataset_size = len(wsi_dataset)
    wsi_dataset_size_indices = list(range(wsi_dataset_size))

    sampler = SubsetRandomSampler(wsi_dataset_size_indices)

    batch_size = 128
    fontsize = 20
    figsize = (60, 20)

    data_loader = DataLoader(
        wsi_dataset,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=True,
        # drop_last=True,
        collate_fn=my_collate,
        # persistent_workers=True,
        num_workers=8)
    
    device = torch.device('cuda')
    model = preact_resnet.PreActResNet50_Ron().to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    start_time = time.time_ns()
    for batch_ndx, batch_data in enumerate(data_loader):
        patches, labels = batch_data
        patches = patches.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(patches)[0][:, 0:(batch_size+1)]

        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()
        
        print(f"batch {batch_ndx} time is {(time.time_ns() - start_time) / (10 ** 9)}")
        start_time = time.time_ns()
        if batch_ndx == 0:
            patches = patches.detach()[:5]
            fig, axs = plt.subplots(ncols=len(patches), squeeze=False)
            for i, img in enumerate(patches):
                img = img.detach()
                img = F.to_pil_image(img)
                axs[0, i].imshow(np.asarray(img))
                axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            plt.savefig('batch.png')
            

