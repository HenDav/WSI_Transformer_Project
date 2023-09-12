from typing import Literal, Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from datasets.datasets import SlideGridDataset
from datasets.features_datasets import SlideGridFeaturesDataset


class WsiGridFeaturesDataModule(LightningDataModule):
    def __init__(
        self,
        features_dir: str,
        grid_side_length: int = 8,
        test_features_dir: Optional[str] = None,
        dataset: str = "CAT",  # TODO: dataset enum/choices
        target: Literal["ER", "PR", "HER2", "KI67"] = "ER",  # TODO: target enum/choices
        val_fold: Optional[int] = 1,
        test_grids_per_slide: int = 1,
        min_tiles_train: int = 100,
        min_tiles_eval: int = 100,
        metadata_file_path: Optional[str] = None,
        batch_size: int = 128,
        num_workers: int = 8,
        **kwargs
    ):
        """
        Args:
            features_dir: path to the features directory
            grid_side_length: side length of the grid
            test_features_dir: path to the test features directory
            dataset: name of the dataset
            target: name of the target to predict
            val_fold: validation fold
            test_grids_per_slide: number of grids to sample from each slide during testing
            min_tiles_train: minimum number of tiles per slide during training
            min_tiles_eval: minimum number of tiles per slide during validation and testing
            metadata_file_path: path to override metadata file
            batch_size: batch size
            num_workers: number of dataloader workers
        """
        super().__init__()

        self.save_hyperparameters()

        self.features_dir = features_dir
        self.grid_side_length = grid_side_length
        self.grid_size = grid_side_length**2
        self.test_features_dir = test_features_dir
        self.dataset = dataset
        self.target = target
        self.val_fold = val_fold
        self.test_grids_per_slide = test_grids_per_slide
        self.min_tiles_train = min_tiles_train
        self.min_tiles_eval = min_tiles_eval
        self.metadata_file_path = metadata_file_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self, *args, **kwargs):
        """Any preparations to be done once before the data is loaded"""
        pass

    def setup(self, stage=None):
        """Initialize datasets / splits and transforms, called on every process in ddp"""

        if stage == "fit":
            self.train_dataset = SlideGridFeaturesDataset(
                features_dir=self.features_dir,
                side_length=self.grid_side_length,
                target=self.target,
                min_tiles=self.min_tiles_train,
                dataset=self.dataset,
                metadata_file_path=self.metadata_file_path,
                train=True,
                val_fold=self.val_fold,
            )

            self.val_dataset = SlideGridFeaturesDataset(
                features_dir=self.features_dir,
                side_length=self.grid_side_length,
                target=self.target,
                min_tiles=self.min_tiles_eval,
                dataset=self.dataset,
                metadata_file_path=self.metadata_file_path,
                train=False,
                val_fold=self.val_fold,
            )
        elif stage == "test":
            self.test_dataset = SlideGridFeaturesDataset(
                features_dir=self.test_features_dir,
                bags_per_slide=self.test_grids_per_slide,
                side_length=self.grid_side_length,
                target=self.target,
                min_tiles=self.min_tiles_eval,
                dataset=self.dataset,
                metadata_file_path=self.metadata_file_path,
                train=False,
                val_fold=self.val_fold,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_grids_per_slide,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class WsiGridDataModule(LightningDataModule):
    def __init__(
        self,
        grid_side_length: int = 8,
        test_features_dir: Optional[str] = None,
        dataset: str = "CAT",  # TODO: dataset enum/choices
        target: Literal["ER", "PR", "HER2", "KI67"] = "ER",  # TODO: target enum/choices
        val_fold: Optional[int] = 1,
        test_grids_per_slide: int = 1,
        batch_size: int = 128,
        num_workers: int = 8,
        **kwargs
    ):
        """
        Args:
            grid_side_length: side length of the grid
            test_features_dir: path to the test features directory
            dataset: name of the dataset
            target: name of the target to predict
            val_fold: validation fold
            test_grids_per_slide: number of grids to sample from each slide during testing
            batch_size: batch size
            num_workers: number of dataloader workers
        """
        super().__init__()

        self.save_hyperparameters()

        self.grid_side_length = grid_side_length
        self.grid_size = grid_side_length**2
        self.test_features_dir = test_features_dir
        self.dataset = dataset
        self.target = target
        self.val_fold = val_fold
        self.test_grids_per_slide = test_grids_per_slide
        self.batch_size = batch_size
        self.num_workers = num_workers

        # TODO: transforms during online feature extraction

    def prepare_data(self, *args, **kwargs):
        """Any preparations to be done once before the data is loaded"""
        pass

    def setup(self, stage=None):
        """Initialize datasets / splits and transforms, called on every process in ddp"""

        if stage == "fit":
            self.train_dataset = SlideGridDataset(
                side_length=self.grid_side_length,
                target=self.target,
                min_tiles=self.min_tiles,
                dataset=self.dataset,
                metadata_file_path=self.metadata_file_path,
                train=True,
                val_fold=self.val_fold,
            )

            self.val_dataset = SlideGridDataset(
                side_length=self.grid_side_length,
                target=self.target,
                min_tiles=self.min_tiles,
                dataset=self.dataset,
                metadata_file_path=self.metadata_file_path,
                train=False,
                val_fold=self.val_fold,
            )
        elif stage == "test":
            self.test_dataset = SlideGridDataset(
                bags_per_slide=self.test_grids_per_slide,
                side_length=self.grid_side_length,
                target=self.target,
                min_tiles=self.min_tiles,
                dataset=self.dataset,
                metadata_file_path=self.metadata_file_path,
                train=False,
                val_fold=self.val_fold,
            )

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
            batch_size=self.test_grids_per_slide,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
