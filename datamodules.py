from argparse import ArgumentParser

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.datasets import WsiDataset  # TODO: update based on dataset api


# TODO: normalization and augmentations for our data
def imagenet_normalization():
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))
    return normalize


class WsiDataModule(LightningDataModule):

    # TODO: update args based on dataset api
    def __init__(self,
                 dataset='TCGA',
                 target='ER',
                 val_fold=1,
                 batch_size=128,
                 num_workers=4,
                 **kwargs):
        super().__init__()

        self.dataset = dataset
        self.target = target
        self.val_fold = val_fold
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.normalization = imagenet_normalization(
        )  # TODO: normalization and augmentations for our data

    def prepare_data(self, *args, **kwargs):
        """Any preparations to be done once before the data is loaded"""

    def setup(self, stage=None):
        """Initialize datasets / splits and transforms, called on every process in ddp"""

        train_transforms = transforms.Compose(
            [transforms.ToTensor(),
             self.normalization()])

        # TODO: update based on dataset api
        self.dataset_train = WsiDataset(dataset=self.dataset,
                                        target=self.target,
                                        val_fold=self.val_fold,
                                        train=True,
                                        transform=train_transforms)

        val_transforms = transforms.Compose(
            [transforms.ToTensor(),
             self.normalization()])

        # TODO: update based on dataset api
        self.dataset_val = WsiDataset(dataset=self.dataset,
                                      target=self.target,
                                      val_fold=self.val_fold,
                                      train=False,
                                      transform=val_transforms)

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        pass
        # TODO: dataloader for evaluation of best model on val at the end of training
        # val_dataloader = self.val_dataloader()
        #
        # return val_dataloader

    @staticmethod
    def add_dataset_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # TODO: update args based on dataset api
        parser.add_argument('--dataset', help='', type=str, default='TCGA')
        parser.add_argument('--target', help='', type=str, default='ER')
        parser.add_argument('--val_fold', help='', type=str, default=1)
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--batch_size", type=int, default=128)

        return parser
