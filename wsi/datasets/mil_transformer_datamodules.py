from typing import Literal, Optional, Dict, Tuple, Callable

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from wsi.datasets.datasets import SlideGridDataset, SlideRandomDataset
from wsi.datasets.features_datasets import SlideGridFeaturesDataset, SlideRandomFeaturesDataset, SlideStridedFeaturesDataset
from torchvision import transforms
from wsi.transformations import MyGaussianNoiseTransform, MyRotation
from wsi.datasets.datamodules import NORMALIZATIONS

class WsiMILDataModule(LightningDataModule):
    def __init__(
        self,
        datasets_folds: Dict,
        datasets_folds_val: Dict,
        target: str,
        bags_per_slide: int,
        min_tiles_train: int,
        min_tiles_eval: int,
        batch_size: int,
        num_workers: int,
        **kwargs
    ):
        """
        Args:
            datasets_folds: A dictionary with dataset names as keys and a list of folds for training as values
            datasets_folds_val: A dictionary with dataset names as keys and a list of folds for validation as values
            target: Name of the target to predict
            bags_per_slide: Number of bags to sample from each slide during training
            min_tiles_train: Minimum number of tiles per slide during training
            min_tiles_eval: Minimum number of tiles per slide during validation and testing
            batch_size: Batch size for the DataLoader
            num_workers: Number of DataLoader workers
        """
        super().__init__()

        self.save_hyperparameters()

        self.datasets_folds = datasets_folds
        self.datasets_folds_val = datasets_folds_val
        self.target = target
        self.bags_per_slide = bags_per_slide
        self.min_tiles_train = min_tiles_train
        self.min_tiles_eval = min_tiles_eval
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self, *args, **kwargs):
        """Any preparations to be done once before the data is loaded"""
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.bags_per_slide,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
class WsiFeaturesDataModule(WsiMILDataModule):
    def __init__(
        self,
        features_dir: str,
        test_features_dir: Optional[str] = None,
        bag_size: int = 64,
        datasets_folds: Dict = {'CAT':[2,3,4,5]},
        datasets_folds_val: Dict = {'CAT':[1]},
        target: str = "er_status",
        bags_per_slide: int = 1,
        min_tiles_train: int = 100,
        min_tiles_eval: int = 100,
        batch_size: int = 128,
        num_workers: int = 8,
        **kwargs
    ):
        """
        Args:
            features_dir: path to the features directory
            test_features_dir: path to the test features directory
            grid_side_length: side length of the grid
            dataset: name of the dataset
            target: name of the target to predict
            val_fold: validation fold
            test_grids_per_slide: number of grids to sample from each slide during testing
            min_tiles_train: minimum number of tiles per slide during training
            min_tiles_eval: minimum number of tiles per slide during validation and testing
            batch_size: batch size
            num_workers: number of dataloader workers
        """
        super().__init__(
            datasets_folds = datasets_folds,
            datasets_folds_val = datasets_folds_val,
            target = target,
            bags_per_slide = bags_per_slide,
            min_tiles_train = min_tiles_train,
            min_tiles_eval = min_tiles_eval,
            batch_size = batch_size,
            num_workers = num_workers
        )

        self.save_hyperparameters()

        self.features_dir = features_dir
        self.grid_side_length = bag_size ** 0.5
        assert self.grid_side_length.is_integer(), f"The square root of the grid size {bag_size} is not a whole number."
        self.grid_side_length = int(self.grid_side_length)
        self.grid_size = bag_size
        self.test_features_dir = test_features_dir

    def setup(self, stage=None):
        """Initialize datasets / splits and transforms, called on every process in ddp"""

        if stage == "fit":
            self.train_dataset = SlideGridFeaturesDataset(
                features_dir=self.features_dir,
                side_length=self.grid_side_length,
                target=self.target,
                min_tiles=self.min_tiles_train,
                datasets_folds=self.datasets_folds,
            )

            self.val_dataset = SlideGridFeaturesDataset(
                features_dir=self.features_dir,
                side_length=self.grid_side_length,
                target=self.target,
                min_tiles=self.min_tiles_eval,
                datasets_folds=self.datasets_folds_val,
            )
        elif stage == "test":
            self.test_dataset = SlideGridFeaturesDataset(
                features_dir=self.test_features_dir,
                bags_per_slide=self.bags_per_slide,
                side_length=self.grid_side_length,
                target=self.target,
                min_tiles=self.min_tiles_eval,
                datasets_folds=self.datasets_folds,
            )

class WsiGridFeaturesDataModule(WsiMILDataModule):
    def __init__(
        self,
        features_dir: str,
        test_features_dir: Optional[str] = None,
        bag_size: int = 64,
        datasets_folds: Dict = {'CAT':[2,3,4,5]},
        datasets_folds_val: Dict = {'CAT':[1]},
        target: str = "er_status",
        bags_per_slide: int = 1,
        min_tiles_train: int = 100,
        min_tiles_eval: int = 100,
        batch_size: int = 128,
        num_workers: int = 8,
        **kwargs
    ):
        """
        Args:
            features_dir: Path to the features directory
            test_features_dir: Path to the test features directory
            bag_size: Number of elements in each bag, the side length of the grid is the square root of this value
            datasets_folds: A dictionary with dataset names as keys and a list of folds for training as values
            datasets_folds_val: A dictionary with dataset names as keys and a list of folds for validation as values
            target: Name of the target to predict
            bags_per_slide: Number of bags to sample from each slide during training
            min_tiles_train: Minimum number of tiles per slide during training
            min_tiles_eval: Minimum number of tiles per slide during validation and testing
            batch_size: Batch size for the DataLoader
            num_workers: Number of DataLoader workers

        Raises:
            AssertionError: If the square root of the bag size is not a whole number.
        """
        super().__init__(
            datasets_folds = datasets_folds,
            datasets_folds_val = datasets_folds_val,
            target = target,
            bags_per_slide = bags_per_slide,
            min_tiles_train = min_tiles_train,
            min_tiles_eval = min_tiles_eval,
            batch_size = batch_size,
            num_workers = num_workers
        )

        self.save_hyperparameters()

        self.features_dir = features_dir
        self.grid_side_length = bag_size ** 0.5
        assert self.grid_side_length.is_integer(), f"The square root of the grid size {bag_size} is not a whole number."
        self.grid_side_length = int(self.grid_side_length)
        self.grid_size = bag_size
        self.test_features_dir = test_features_dir

    def setup(self, stage=None):
        """Initialize datasets / splits and transforms, called on every process in ddp"""

        if stage == "fit":
            self.train_dataset = SlideGridFeaturesDataset(
                features_dir=self.features_dir,
                side_length=self.grid_side_length,
                target=self.target,
                min_tiles=self.min_tiles_train,
                datasets_folds=self.datasets_folds,
            )

            self.val_dataset = SlideGridFeaturesDataset(
                features_dir=self.features_dir,
                side_length=self.grid_side_length,
                target=self.target,
                min_tiles=self.min_tiles_eval,
                datasets_folds=self.datasets_folds_val,
            )
        elif stage == "test":
            self.test_dataset = SlideGridFeaturesDataset(
                features_dir=self.test_features_dir,
                bags_per_slide=self.bags_per_slide,
                side_length=self.grid_side_length,
                target=self.target,
                min_tiles=self.min_tiles_eval,
                datasets_folds=self.datasets_folds,
            )


class WsiRandomFeaturesDataModule(WsiMILDataModule):
    def __init__(
        self,
        features_dir: str,
        test_features_dir: Optional[str] = None,
        bag_size: int = 100,
        datasets_folds: Dict = {'CAT':[2,3,4,5]},
        datasets_folds_val: Dict = {'CAT':[1]},
        target: str = "er_status",
        bags_per_slide: int = 1,
        min_tiles_train: int = 100,
        min_tiles_eval: int = 100,
        batch_size: int = 128,
        num_workers: int = 8,
        **kwargs
    ):
        """
        Args:
            features_dir: Path to the features directory
            test_features_dir: Path to the test features directory
            bag_size: Number of elements in each bag, the side length of the grid is the square root of this value
            datasets_folds: A dictionary with dataset names as keys and a list of folds for training as values
            datasets_folds_val: A dictionary with dataset names as keys and a list of folds for validation as values
            target: Name of the target to predict
            bags_per_slide: Number of bags to sample from each slide during training
            min_tiles_train: Minimum number of tiles per slide during training
            min_tiles_eval: Minimum number of tiles per slide during validation and testing
            batch_size: Batch size for the DataLoader
            num_workers: Number of DataLoader workers
        """
        super().__init__(
            datasets_folds = datasets_folds,
            datasets_folds_val = datasets_folds_val,
            target = target,
            bags_per_slide = bags_per_slide,
            min_tiles_train = min_tiles_train,
            min_tiles_eval = min_tiles_eval,
            batch_size = batch_size,
            num_workers = num_workers
        )

        self.save_hyperparameters()

        self.features_dir = features_dir
        self.bag_size = bag_size
        self.test_features_dir = test_features_dir

    def setup(self, stage=None):
        """Initialize datasets / splits and transforms, called on every process in ddp"""

        if stage == "fit":
            self.train_dataset = SlideRandomFeaturesDataset(
                features_dir=self.features_dir,
                bag_size=self.bag_size,
                target=self.target,
                min_tiles=self.min_tiles_train,
                datasets_folds=self.datasets_folds,
            )

            self.val_dataset = SlideRandomFeaturesDataset(
                bags_per_slide=self.bags_per_slide,
                features_dir=self.test_features_dir,
                bag_size=self.bag_size,
                target=self.target,
                min_tiles=self.min_tiles_eval,
                datasets_folds=self.datasets_folds_val,
            )
        elif stage == "test":
            self.test_dataset = SlideRandomFeaturesDataset(
                features_dir=self.test_features_dir,
                bags_per_slide=self.bags_per_slide,
                bag_size=self.bag_size,
                target=self.target,
                min_tiles=self.min_tiles_eval,
                datasets_folds=self.datasets_folds,
            )


class WsiRandomDataModule(WsiMILDataModule):
    def __init__(
        self,
        bag_size: int = 64,
        datasets_folds: Dict = {'CAT':[2,3,4,5]},
        datasets_folds_val: Dict = {'CAT':[1]},
        min_tiles_train: int = 10,
        min_tiles_eval: int = 10,
        img_size: int = 256,
        target: str = "er_status",
        bags_per_slide: int = 1,
        batch_size: int = 128,
        num_workers: int = 8,
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
            grid_side_length: side length of the grid
            dataset: name of the dataset
            target: name of the target to predict
            val_fold: validation fold
            test_grids_per_slide: number of grids to sample from each slide during testing
            batch_size: batch size
            num_workers: number of dataloader workers
        """
        super().__init__(
            datasets_folds = datasets_folds,
            datasets_folds_val = datasets_folds_val,
            target = target,
            bags_per_slide = bags_per_slide,
            min_tiles_train = min_tiles_train,
            min_tiles_eval = min_tiles_eval,
            batch_size = batch_size,
            num_workers = num_workers
        )
        
        self.save_hyperparameters()

        self.grid_side_length = bag_size ** 0.5
        assert self.grid_side_length.is_integer(), f"The square root of the grid size {bag_size} is not a whole number."
        self.grid_side_length = int(self.grid_side_length)
        self.img_size = img_size * self.grid_side_length
        self.bag_size = bag_size
        self.test_grids_per_slide = bags_per_slide
        self.autoaug = autoaug
        self.normalization = normalization
        self.openslide = openslide

        self.GIPDEEP10_OPENSLIDE_ROOT = "/data"
        self.GIPDEEP10_H5_ROOT = "/data/unsynced_data/h5"

        if ssd:
            self.GIPDEEP10_OPENSLIDE_ROOT = "/SSDStorage"
            self.GIPDEEP10_H5_ROOT = "/SSDStorage/h5"

        self.train_transforms, self.eval_transforms = (
            self.define_transforms() if transforms is None else transforms
        )

    def setup(self, stage=None):
        """Initialize datasets / splits and transforms, called on every process in ddp"""

        if stage == "fit":
            self.train_dataset = SlideRandomDataset(
                bag_size=self.bag_size,
                target=self.target,
                min_tiles=self.min_tiles_train,
                datasets_folds=self.datasets_folds,
                transform=self.train_transforms,
                datasets_base_dir_path=(
                    self.GIPDEEP10_OPENSLIDE_ROOT if self.openslide else self.GIPDEEP10_H5_ROOT
                ),
            )

            self.val_dataset = SlideRandomDataset(
                bag_size=self.bag_size,
                bags_per_slide=self.bags_per_slide,
                target=self.target,
                min_tiles=self.min_tiles_eval,
                transform=self.eval_transforms,
                datasets_folds=self.datasets_folds_val,
                datasets_base_dir_path=(
                    self.GIPDEEP10_OPENSLIDE_ROOT if self.openslide else self.GIPDEEP10_H5_ROOT
                ),
            )
        elif stage == "test":
            self.test_dataset = SlideRandomDataset(
                bag_size=self.bag_size,
                bags_per_slide=self.bags_per_slide,
                target=self.target,
                min_tiles=self.min_tiles_eval,
                transform=self.eval_transforms,
                datasets_folds=self.datasets_folds,
                datasets_base_dir_path=(
                    self.GIPDEEP10_OPENSLIDE_ROOT if self.openslide else self.GIPDEEP10_H5_ROOT
                ),
            )

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
                    # transforms.GaussianBlur(3, sigma=(1e-7, 1e-1)), # doesn't make sense in random bag with bag transform
                    MyGaussianNoiseTransform(sigma=(0, 0.05)),
                    # transforms.RandomAffine(degrees=0, scale=(1, 1 + scale_factor)), # doesn't make sense in random bag with bag transform
                ]
            train_transforms = [
                *transform_ron,
                *train_transforms,
            ]

        train_transforms = [
            transforms.RandomCrop(size=self.img_size),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            *train_transforms,
        ]

        train_transforms = transforms.Compose(train_transforms)
        eval_transforms = transforms.Compose(eval_transforms)

        return train_transforms, eval_transforms


class WsiGridDataModule(WsiMILDataModule):
    def __init__(
        self,
        bag_size: int = 64,
        datasets_folds: Dict = {'CAT':[2,3,4,5]},
        datasets_folds_val: Dict = {'CAT':[1]},
        min_tiles_train: int = 10,
        min_tiles_eval: int = 10,
        img_size: int = 256,
        target: str = "er_status",
        bags_per_slide: int = 1,
        batch_size: int = 128,
        num_workers: int = 8,
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
            grid_side_length: side length of the grid
            dataset: name of the dataset
            target: name of the target to predict
            val_fold: validation fold
            test_grids_per_slide: number of grids to sample from each slide during testing
            batch_size: batch size
            num_workers: number of dataloader workers
        """
        super().__init__(
            datasets_folds = datasets_folds,
            datasets_folds_val = datasets_folds_val,
            target = target,
            bags_per_slide = bags_per_slide,
            min_tiles_train = min_tiles_train,
            min_tiles_eval = min_tiles_eval,
            batch_size = batch_size,
            num_workers = num_workers
        )
        
        self.save_hyperparameters()

        self.grid_side_length = bag_size ** 0.5
        assert self.grid_side_length.is_integer(), f"The square root of the grid size {bag_size} is not a whole number."
        self.grid_side_length = int(self.grid_side_length)
        self.img_size = img_size * self.grid_side_length
        self.bags_per_slide = bags_per_slide
        self.autoaug = autoaug
        self.normalization = normalization
        self.openslide = openslide

        self.GIPDEEP10_OPENSLIDE_ROOT = "/data"
        self.GIPDEEP10_H5_ROOT = "/data/unsynced_data/h5"

        if ssd:
            self.GIPDEEP10_OPENSLIDE_ROOT = "/SSDStorage"
            self.GIPDEEP10_H5_ROOT = "/SSDStorage/h5"

        self.train_transforms, self.eval_transforms = (
            self.define_transforms() if transforms is None else transforms
        )

    def setup(self, stage=None):
        """Initialize datasets / splits and transforms, called on every process in ddp"""

        if stage == "fit":
            self.train_dataset = SlideGridDataset(
                side_length=self.grid_side_length,
                target=self.target,
                min_tiles=self.min_tiles_train,
                datasets_folds=self.datasets_folds,
                transform=self.train_transforms,
                datasets_base_dir_path=(
                    self.GIPDEEP10_OPENSLIDE_ROOT if self.openslide else self.GIPDEEP10_H5_ROOT
                ),
            )

            self.val_dataset = SlideGridDataset(
                bags_per_slide=self.bags_per_slide,
                side_length=self.grid_side_length,
                target=self.target,
                min_tiles=self.min_tiles_eval,
                transform=self.eval_transforms,
                datasets_folds=self.datasets_folds_val,
                datasets_base_dir_path=(
                    self.GIPDEEP10_OPENSLIDE_ROOT if self.openslide else self.GIPDEEP10_H5_ROOT
                ),
            )
        elif stage == "test":
            self.test_dataset = SlideGridDataset(
                bags_per_slide=self.bags_per_slide,
                side_length=self.grid_side_length,
                target=self.target,
                min_tiles=self.min_tiles_eval,
                transform=self.eval_transforms,
                datasets_folds=self.datasets_folds,
                datasets_base_dir_path=(
                    self.GIPDEEP10_OPENSLIDE_ROOT if self.openslide else self.GIPDEEP10_H5_ROOT
                ),
            )

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
                    transforms.RandomAffine(degrees=0, scale=(1, 1 + scale_factor)),
                ]
            train_transforms = [
                *transform_ron,
                *train_transforms,
            ]

        train_transforms = [
            transforms.RandomCrop(size=self.img_size),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            *train_transforms,
        ]

        train_transforms = transforms.Compose(train_transforms)
        eval_transforms = transforms.Compose(eval_transforms)

        return train_transforms, eval_transforms


