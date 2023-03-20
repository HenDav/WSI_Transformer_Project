from typing import Callable, Literal, Optional, Tuple

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.datasets import (
    RandomPatchDataset,
    SerialPatchDataset,
    SlideStridedDataset,
)

# from legacy.datasets import WSI_REGdataset

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

GIPDEEP10_DATA_ROOT = "/data"


class WsiDataModule(LightningDataModule):
    def __init__(
        self,
        dataset: str = "CAT",  # TODO: dataset enum/choices
        target: str = "ER",  # TODO: target enum/choices
        val_fold: Optional[int] = 1,
        eval_on_train: bool = False,
        patches_per_slide_train: int = 10,
        patches_per_slide_eval: int = 10,
        min_tiles_eval: int = 100,
        img_size: int = 256,
        batch_size: int = 256,
        num_workers: int = 8,
        normalization: Literal[
            "standard", "imagenet", "wsi_ron", "tcga", "cat", "none"
        ] = "cat",
        autoaug: bool = False,
        transforms: Optional[Tuple[Callable, Callable]] = None,
        openslide: bool = False,
        **kwargs
    ):
        """
        Args:
            dataset: name of the dataset
            target: name of the target to predict
            val_fold: validation fold
            eval_on_train: whether to evaluate on the training set for test/predict
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

        self.dataset = dataset
        self.target = target
        self.val_fold = val_fold
        self.eval_on_train = eval_on_train
        self.patches_per_slide_train = patches_per_slide_train
        self.patches_per_slide_eval = patches_per_slide_eval
        self.min_tiles_eval = min_tiles_eval
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalization = normalization
        self.autoaug = autoaug
        self.openslide = openslide

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
                min_tiles=self.patches_per_slide_train,
                train=True,
                transform=self.train_transforms,
                datasets_base_dir_path=(
                    GIPDEEP10_DATA_ROOT if self.openslide else None
                ),
            )

            self.val_dataset = SlideStridedDataset(
                dataset=self.dataset,
                target=self.target,
                val_fold=self.val_fold,
                bag_size=self.patches_per_slide_eval,
                min_tiles=self.min_tiles_eval,
                train=False,
                transform=self.eval_transforms,
                datasets_base_dir_path=(
                    GIPDEEP10_DATA_ROOT if self.openslide else None
                ),
            )
        elif stage == "test":
            # TODO: allow testing on more patches per slide with batching, currently the bag size is the batch size so this is limited
            self.test_dataset = SlideStridedDataset(
                dataset=self.dataset,
                target=self.target,
                val_fold=self.val_fold,
                bag_size=self.patches_per_slide_eval,
                min_tiles=self.min_tiles_eval,
                train=self.eval_on_train,
                transform=self.eval_transforms,
                datasets_base_dir_path=(
                    GIPDEEP10_DATA_ROOT if self.openslide else None
                ),
            )
        elif stage == "predict":
            self.predict_dataset = SerialPatchDataset(
                dataset=self.dataset,
                target=self.target,
                val_fold=self.val_fold,
                min_tiles=self.min_tiles_eval,
                train=self.eval_on_train,
                transform=self.eval_transforms,
                datasets_base_dir_path=(
                    GIPDEEP10_DATA_ROOT if self.openslide else None
                ),
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=None,  # each instance is a bag
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
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

        if self.autoaug:
            train_transforms = [
                transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
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


# class LegacyWsiDataModule(LightningDataModule):
#     def __init__(
#         self,
#         dataset: str = "CAT",  # TODO: dataset enum/choices
#         target: Literal["ER", "PR", "HER2", "KI67"] = "ER",  # TODO: target enum/choices
#         val_fold: Optional[int] = 1,
#         eval_on_train: bool = False,
#         patches_per_slide_train: int = 10,
#         patches_per_slide_eval: int = 10,
#         img_size: int = 256,
#         batch_size: int = 32,
#         num_workers: int = 8,
#         normalization: Literal[
#             "standard", "imagenet", "wsi_ron", "tcga", "cat", "none"
#         ] = "cat",
#         autoaug: bool = False,
#         **kwargs
#     ):
#         """
#         Args:
#             dataset: name of the dataset
#             target: name of the target to predict
#             val_fold: validation fold
#             eval_on_train: whether to evaluate on the training set for test/predict
#             patches_per_slide_train: number of patches per slide in each training epoch
#             patches_per_slide_eval: number of patches per slide in evaluation
#             img_size: square image dimension at entry to the model
#             batch_size: batch size for training and feature extraction
#             num_workers: number of dataloader workers
#             normalization: normalization scheme
#             autoaug: whether to use autoaugment imagenet recipe for default train transforms
#             transforms: override default transforms, of the form (train_transforms, eval_transforms)
#         """
#         super().__init__()
#
#         self.save_hyperparameters()
#
#         self.dataset = dataset
#         self.target = target
#         self.val_fold = val_fold
#         self.eval_on_train = eval_on_train
#         self.patches_per_slide_train = patches_per_slide_train
#         self.patches_per_slide_eval = patches_per_slide_eval
#         self.img_size = img_size
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.normalization = normalization
#         self.autoaug = autoaug
#
#     def prepare_data(self, *args, **kwargs):
#         """Any preparations to be done once before the data is loaded"""
#         pass
#
#     def setup(self, stage=None):
#         """Initialize datasets / splits and transforms, called on every process in ddp"""
#
#         if stage == "fit":
#             self.train_dataset = WSI_REGdataset(
#                 DataSet=self.dataset,
#                 target_kind=self.target,
#                 test_fold=self.val_fold,
#                 train=True,
#                 transform_type="pcbnfrsc",
#                 n_tiles=self.patches_per_slide_train,
#             )
#             self.val_dataset = WSI_REGdataset(
#                 DataSet=self.dataset,
#                 target_kind=self.target,
#                 test_fold=self.val_fold,
#                 train=False,
#                 transform_type="none",
#                 n_tiles=self.patches_per_slide_eval,
#             )
#
#     def train_dataloader(self):
#         return DataLoader(
#             self.train_dataset,
#             batch_size=self.batch_size,
#             shuffle=True,
#             num_workers=self.num_workers,
#             pin_memory=True,
#             drop_last=True,
#         )
#
#     def val_dataloader(self):
#         return DataLoader(
#             self.val_dataset,
#             batch_size=self.batch_size,
#             shuffle=False,
#             num_workers=self.num_workers,
#             pin_memory=True,
#         )
