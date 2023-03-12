import time
from datetime import datetime
from pathlib import Path

import matplotlib.lines

# matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# pandas
import pandas

# torch
import torch
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler, SubsetRandomSampler
from torchvision import transforms

from datasets.datasets import SlideStridedDataset
from models import preact_resnet

# gipmed
# from nn.experiments import Experiment, ExperimentArgumentsParser
from wsi_core import constants
from wsi_core.wsi import BioMarker

# plt.rcParams["savefig.bbox"] = 'tight'


if __name__ == "__main__":
    wsi_dataset = SlideStridedDataset(
        metadata_file_path="/home/dahen/WSI/metadata.csv",
        bag_size=32,
        transform=F.to_tensor,
    )

    fontsize = 20
    figsize = (60, 20)

    data_loader = DataLoader(
        wsi_dataset,
        batch_size=None,
        pin_memory=True,
        # drop_last=True,
        # persistent_workers=True,
        num_workers=0,
    )

    device = torch.device("cuda")
    model = preact_resnet.PreActResNet50_Ron().to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    start_time = time.time_ns()
    for batch_ndx, item in enumerate(data_loader):
        bag = item["bag"]
        labels = item["label"]

        bag = bag.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(bag)[0]

        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()

        print(f"batch {batch_ndx} time is {(time.time_ns() - start_time) / (10 ** 9)}")
        start_time = time.time_ns()
        if batch_ndx == 0:
            bag = bag.detach()
            fig, axs = plt.subplots(ncols=len(bag), squeeze=False)
            for i, img in enumerate(bag):
                img = img.detach()
                img = F.to_pil_image(img)
                axs[0, i].imshow(np.asarray(img))
                axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            plt.savefig("batch.png")
