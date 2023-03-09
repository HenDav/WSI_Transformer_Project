from typing import Callable, Literal, Optional, Tuple

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.datasets import (
    RandomPatchDataset,
    SerialPatchDataset,
    SlideStridedDataset,
)

NORMALIZATIONS = {
    "standard": {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
    "imagenet": {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
    "wsi_ron": {"mean": [0.8998, 0.8253, 0.9357], "std": [0.1125, 0.1751, 0.0787]},
    "tcga": {
        "mean": [58.2069073 / 255, 96.22645279 / 255, 70.26442606 / 255],
        "std": [
            40.40400300279664 / 255,
            58.90625962739444 / 255,
            45.09334057330417 / 255,
        ],
    },
    "none": {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
}


class WsiDataModule(LightningDataModule):
    def __init__(
        self,
        dataset: Literal["CAT"] = "CAT",  # TODO: dataset enum/choices
        target: Literal["ER", "PR", "Her2", "Ki67"] = "ER",  # TODO: target enum/choices
        val_fold: Optional[int] = 1,
        patches_per_slide_train: int = 10,
        patches_per_slide_eval: int = 100,
        img_size: int = 256,
        batch_size: int = 128,
        num_workers: int = 8,
        normalization: Literal[
            "standard", "imagenet", "wsi_ron", "tcga", "none"
        ] = "none",
        transforms: Optional[Tuple[Callable, Callable]] = None,
        **kwargs
    ):
        """
        Args:
            dataset: name of the dataset
            target: name of the target to predict
            val_fold: validation fold
            patches_per_slide_train: number of patches per slide in each training epoch
            patches_per_slide_eval: number of patches per slide in evaluation
            img_size: square image dimension at entry to the model
            batch_size: batch size for training and feature extraction
            num_workers: number of dataloader workers
            normalization: normalization scheme
            transforms: override default transforms, of the form (train_transforms, eval_transforms)
        """
        super().__init__()

        self.save_hyperparameters()

        self.dataset = dataset
        self.target = target
        self.val_fold = val_fold
        self.patches_per_slide_train = patches_per_slide_train
        self.patches_per_slide_eval = patches_per_slide_eval
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalization = normalization

        self.train_transforms, self.eval_transforms = (
            self.define_transforms() if transforms is None else transforms
        )

    def prepare_data(self, *args, **kwargs):
        """Any preparations to be done once before the data is loaded"""
        pass

    def setup(self, stage=None):
        """Initialize datasets / splits and transforms, called on every process in ddp"""

        if stage == "fit":
            self.train_dataset = RandomPatchDataset(
                dataset=self.dataset,
                target=self.target,
                val_fold=self.val_fold,
                patches_per_slide=self.patches_per_slide_train,
                train=True,
                transform=self.train_transforms,
            )

            self.val_dataset = SlideStridedDataset(
                dataset=self.dataset,
                target=self.target,
                val_fold=self.val_fold,
                bag_size=self.patches_per_slide_eval,
                train=False,
                transform=self.eval_transforms,
            )
        elif stage == "test":
            # TODO: allow testing on more patches per slide with batching, currently the bag size is the batch size so this is limited
            self.test_dataset = SlideStridedDataset(
                dataset=self.dataset,
                target=self.target,
                val_fold=self.val_fold,
                bag_size=self.patches_per_slide_eval,
                train=False,
                transform=self.eval_transforms,
            )
        elif stage == "predict":
            self.predict_dataset = SerialPatchDataset(
                dataset=self.dataset,
                target=self.target,
                val_fold=self.val_fold,
                train=False,
                transform=self.eval_transforms,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=None,  # each instance is a bag
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=None,  # each instance is a bag
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def define_transforms(self):
        normalization = transforms.Normalize(**NORMALIZATIONS[self.normalization])

        train_transforms = [transforms.ToTensor(), normalization]
        eval_transforms = [
            transforms.CenterCrop(size=self.img_size),
            transforms.ToTensor(),
            normalization,
        ]

        train_transforms = [
            transforms.RandomResizedCrop(size=self.img_size),
            transforms.RandomHorizontalFlip(),
            *train_transforms,
        ]

        train_transforms = transforms.Compose(train_transforms)
        eval_transforms = transforms.Compose(eval_transforms)

        return train_transforms, eval_transforms
