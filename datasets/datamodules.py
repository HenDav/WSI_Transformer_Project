from typing import Callable, Literal, Optional, Tuple, Dict

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.datasets import (
    RandomPatchDataset,
    SerialPatchDataset,
    SlideStridedDataset,
    SlideRandomDataset,
)

from legacy.transformations import MyGaussianNoiseTransform, MyRotation
# from legacy.datasets_legacy import WSI_REGdataset

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
    "cat": {
        "mean": [0.8248, 0.6225, 0.7605],
        "std": [0.1730, 0.2398, 0.1740],
    },
    "none": {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
}

class WsiDataModule(LightningDataModule):
    def __init__(
        self,
        datasets_folds: Dict = {"CAT": [2,3,4,5]},
        datasets_folds_val: Dict = {"CAT": [1]},
        target: str = "ER",  # TODO: target enum/choices
        patches_per_slide_train: int = 10,
        patches_per_slide_eval: int = 10,
        min_tiles_eval: int = 100,
        img_size: int = 256,
        batch_size: int = 256,
        num_workers: int = 12,
        normalization: Literal[
            "standard", "imagenet", "wsi_ron", "tcga", "cat", "none"
        ] = "cat",
        autoaug: Literal[
            "imagenet", "wsi_ron", "none"
        ] = "imagenet",
        transforms: Optional[Tuple[Callable, Callable]] = None,
        openslide: bool = False,
        ssd: bool = True,
        **kwargs
    ):
        """
        Args:
            dataset: name of the dataset
            target: name of the target to predict
            datasets_folds: datasets and folds to train/test on
            datasets_folds_val: datasets and folds to preform validation on while training
            patches_per_slide_train: number of patches per slide in each training epoch
            patches_per_slide_eval: number of patches per slide in evaluation
            min_tiles_test: minimum number of tiles per slide
            img_size: square image dimension at entry to the model
            batch_size: batch size for training and feature extraction
            num_workers: number of dataloader workers
            normalization: normalization scheme
            autoaug: whether to use autoaugment imagenet recipe for default train transforms
            transforms: override default transforms, of the form (train_transforms, eval_transforms)
            openslide: whether to use openslide for reading images
        """
        super().__init__()

        self.save_hyperparameters()

        self.datasets_folds = datasets_folds
        self.datasets_folds_val = datasets_folds_val
        self.target = target
        self.patches_per_slide_train = patches_per_slide_train
        self.patches_per_slide_eval = patches_per_slide_eval
        self.min_tiles_eval = min_tiles_eval
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalization = normalization
        self.autoaug = autoaug
        self.openslide = openslide

        self.GIPDEEP10_OPENSLIDE_ROOT = "/data"
        self.GIPDEEP10_H5_ROOT = "/data/unsynced_data/h5"

        if ssd:
            self.GIPDEEP10_OPENSLIDE_ROOT = "/SSDStorage"
            self.GIPDEEP10_H5_ROOT = "/SSDStorage/h5"

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
                datasets_folds=self.datasets_folds,
                target=self.target,
                patches_per_slide=self.patches_per_slide_train,
                min_tiles=self.patches_per_slide_train,
                transform=self.train_transforms,
                datasets_base_dir_path=(
                    self.GIPDEEP10_OPENSLIDE_ROOT if self.openslide else self.GIPDEEP10_H5_ROOT
                ),
            )

            self.val_dataset = SlideStridedDataset(
                datasets_folds=self.datasets_folds_val,
                target=self.target,
                bag_size=self.patches_per_slide_eval,
                min_tiles=self.min_tiles_eval,
                transform=self.eval_transforms,
                datasets_base_dir_path=(
                    self.GIPDEEP10_OPENSLIDE_ROOT if self.openslide else self.GIPDEEP10_H5_ROOT
                ),
            )
            
            self.train_dloader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True,
                prefetch_factor=10,
                persistent_workers=True,
            )
            
            self.val_dloader = DataLoader(
                self.val_dataset,
                batch_size=None,  # each instance is a bag
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                prefetch_factor=10,
                persistent_workers=True,
            )
            
        elif stage == "test":
            # TODO: allow testing on more patches per slide with batching, currently the bag size is the batch size so this is limited
            self.test_dataset = SlideStridedDataset(
                datasets_folds=self.datasets_folds,
                target=self.target,
                val_fold=self.val_fold,
                bag_size=self.patches_per_slide_eval,
                min_tiles=self.min_tiles_eval,
                transform=self.eval_transforms,
                datasets_base_dir_path=(
                    self.GIPDEEP10_OPENSLIDE_ROOT if self.openslide else self.GIPDEEP10_H5_ROOT
                ),
            )
            self.test_dloader = DataLoader(
                    self.test_dataset,
                    batch_size=None,  # each instance is a bag
                    shuffle=False,
                    num_workers=self.num_workers,
                    prefetch_factor=10,
                    pin_memory=True,
                )
            
        elif stage == "predict":
            self.predict_dataset = SerialPatchDataset(
                dataset=self.dataset,
                target=self.target,
                val_fold=self.val_fold,
                min_tiles=self.min_tiles_eval,
                transform=self.eval_transforms,
                datasets_base_dir_path=(
                    self.GIPDEEP10_OPENSLIDE_ROOT if self.openslide else self.GIPDEEP10_H5_ROOT
                ),
            )
            
            self.predict_dloader = DataLoader(
                    self.predict_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    prefetch_factor=10,
                    pin_memory=True,
            )

    def train_dataloader(self):
        return self.train_dloader

    def val_dataloader(self):
        return self.val_dloader

    def test_dataloader(self):
        return self.test_dloader

    def predict_dataloader(self):
        return self.predict_dloader

    def define_transforms(self):
        normalization = transforms.Normalize(**NORMALIZATIONS[self.normalization])

        train_transforms = [transforms.ToTensor(), normalization]
        eval_transforms = [
            transforms.CenterCrop(size=self.img_size),
            transforms.ToTensor(),
            normalization,
        ]

        if self.autoaug == "imagenet":
            train_transforms = [
                transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
                *train_transforms,
            ]
            
        if self.autoaug == "wsi_ron":
            color_param = 0.1
            scale_factor = 0.2
            transform_ron = \
                [
                    transforms.ColorJitter(brightness=color_param, contrast=color_param * 2,
                                           saturation=color_param, hue=color_param),
                    transforms.GaussianBlur(3, sigma=(1e-7, 1e-1)),
                    MyGaussianNoiseTransform(sigma=(0, 0.05)),
                    transforms.RandomVerticalFlip(),
                    MyRotation(angles=[0, 90, 180, 270]),
                    transforms.RandomAffine(degrees=0, scale=(1, 1 + scale_factor)),
                ]
            train_transforms = [
                *transform_ron,
                *train_transforms,
            ]

        train_transforms = [
            transforms.RandomCrop(size=self.img_size),
            transforms.RandomHorizontalFlip(),
            *train_transforms,
        ]

        train_transforms = transforms.Compose(train_transforms)
        eval_transforms = transforms.Compose(eval_transforms)

        return train_transforms, eval_transforms