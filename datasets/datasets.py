import socket
from abc import ABC, abstractmethod
from typing import List, Optional

import pandas
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets.slides_manager import SlidesManager
from wsi_core import constants
from wsi_core.base import SeedableObject
from wsi_core.wsi import (
    BioMarker,
    GridPatchExtractor,
    RandomPatchExtractor,
    SerialPatchExtractor,
    Slide,
    StridedPatchExtractor,
)


class WSIDataset(ABC, Dataset, SeedableObject):
    def __init__(
        self,
        instances_per_slide: int = 10,
        target: str = "ER",
        tile_size: int = 256,
        desired_mpp: float = 1.0,
        metadata_at_magnification: int = 10,
        min_tiles: int = 100,
        dataset: str = "CAT",
        metadata_file_path: Optional[str] = None,
        datasets_base_dir_path: Optional[str] = None,
        transform=transforms.Compose([]),
        val_fold: Optional[int] = None,
        train: bool = True,
        slides_manager: SlidesManager = None,
        **kw: object
    ):
        super().__init__(**kw)
        if not slides_manager:
            if not datasets_base_dir_path:
                if socket.gethostname() == "gipdeep10":
                    datasets_base_dir_path = constants.data_root_gipdeep10
                else:
                    datasets_base_dir_path = constants.data_root_netapp
            assert (
                len(datasets_base_dir_path) > 0
            ), "Problem with datasets_base_dir_path path"
            if not metadata_file_path:
                metadata_file_path = constants.main_metadata_csv
            datasets = constants.get_dataset_ids(dataset)
            current_folds = constants.folds_for_datasets[next(iter(datasets))]
            if val_fold is not None and train:
                current_folds.remove(val_fold)
            elif val_fold is not None and not train:
                current_folds = [val_fold]
            slides_manager = SlidesManager(
                datasets_base_dir_path=datasets_base_dir_path,
                desired_mpp=desired_mpp,
                tile_size=tile_size,
                metadata_at_magnification=metadata_at_magnification,
                metadata_file_path=metadata_file_path,
                row_predicate=self.default_predicate,
                min_tiles=min_tiles,
                datasets=datasets,
                folds=current_folds,
            )
        self._slides_manager = slides_manager
        self.num_slides = len(slides_manager)
        self._dataset_size = instances_per_slide * self.num_slides

    def __len__(self):
        return self._dataset_size

    # @abstractmethod
    # def __getitem__(self, item: int) -> object:
    #     pass

    def set_folds(self, folds: List[int]):
        self._slides_manager.filter_folds(folds=folds)

    @staticmethod
    def default_predicate(
        df: pandas.DataFrame = None,
        min_tiles: int = 100,
        datasets: List[str] = ["CARMEL1"],
        folds: List[int] = constants.folds_for_datasets["CARMEL1"],
    ) -> pandas.Index:
        fold_indecies = df.index[df[constants.fold_column_name].isin(folds)]
        dataset_indecies = df.index[df[constants.dataset_id_column_name].isin(datasets)]
        min_tiles_indecies = df.index[
            df[constants.legitimate_tiles_column_name] > min_tiles
        ]
        return min_tiles_indecies.intersection(dataset_indecies).intersection(
            fold_indecies
        )


class RandomPatchDataset(WSIDataset):
    def __init__(
        self,
        patches_per_slide: int = 10,
        target: str = "ER",
        tile_size: int = 256,  # TODO: make this useful through patch extraction
        desired_mpp: float = 1.0,  # TODO: make this useful through patch extraction
        metadata_at_magnification: int = 10,
        min_tiles: int = 100,
        dataset: str = "CAT",
        metadata_file_path: str = None,
        datasets_base_dir_path: str = None,
        transform=transforms.Compose([]),
        train: bool = True,  # controls folds
        val_fold: int = None,
        slides_manager: SlidesManager = None,
        **kw: object
    ):
        super().__init__(
            instances_per_slide=patches_per_slide,
            slides_manager=slides_manager,
            target=target,
            tile_size=tile_size,
            desired_mpp=desired_mpp,
            metadata_at_magnification=metadata_at_magnification,
            min_tiles=min_tiles,
            dataset=dataset,
            metadata_file_path=metadata_file_path,
            datasets_base_dir_path=datasets_base_dir_path,
            transform=transform,
            val_fold=val_fold,
            train=train,
            **kw
        )
        self._transform = transform
        self._target = BioMarker[target]
        self._train = train

    def __getitem__(self, item: int):
        slide = self._slides_manager.get_slide(item % self.num_slides)

        slide_name = slide.slide_context.image_file_name_stem
        patch_extractor = RandomPatchExtractor(slide=slide)
        patch, center_pixel = patch_extractor.extract_patch(patch_validators=[])
        label = slide.slide_context.get_biomarker_value(bio_marker=self._target)
        if (label == "Positive"):
            label = 1
        elif (label == "Negative"):
            label = 0

        return {
            "patch": self._transform(transforms.ToTensor()(patch.image)),
            "label": label,
            "slide_name": slide_name,
            "center_pixel": center_pixel,
        }


class SerialPatchDataset(WSIDataset):
    def __init__(
        self,
        patches_per_slide: int = 10,  # TODO: remove this and update len of the dataset
        target: str = "ER",
        tile_size: int = 256,  # TODO: make this useful through patch extraction
        desired_mpp: float = 1.0,  # TODO: make this useful through patch extraction
        metadata_at_magnification: int = 10,
        min_tiles: int = 100,
        dataset: str = "CAT",
        metadata_file_path: str = None,
        datasets_base_dir_path: str = None,
        transform=transforms.Compose([]),
        train: bool = True,  # controls folds
        val_fold: int = None,
        slides_manager: SlidesManager = None,
        **kw: object
    ):
        super().__init__(
            instances_per_slide=patches_per_slide,
            slides_manager=slides_manager,
            target=target,
            tile_size=tile_size,
            desired_mpp=desired_mpp,
            metadata_at_magnification=metadata_at_magnification,
            min_tiles=min_tiles,
            dataset=dataset,
            metadata_file_path=metadata_file_path,
            datasets_base_dir_path=datasets_base_dir_path,
            transform=transform,
            val_fold=val_fold,
            train=train,
            **kw
        )
        self._transform = transform
        self._target = BioMarker[target]
        self._train = train
        self._n_slides = 1  # slides touched so far
        self._slide = self._slides_manager.get_slide(self._n_slides - 1)
        self._n_slide_patches = 0
        self.patch_extractor = SerialPatchExtractor(slide=self._slide)

    def __getitem__(self, item: int):
        if self._n_slide_patches >= (self._slide.tiles_count - 1):
            self._n_slides += 1
            self._slide = self._slides_manager.get_slide(self._n_slides - 1)
            self._n_slide_patches = 0
        self._n_slide_patches += 1
        slide_name = self._slide.slide_context.image_file_name_stem
        patch, center_pixel = self.patch_extractor.extract_patch(patch_validators=[])
        label = self._slide.slide_context.get_biomarker_value(bio_marker=self._target)
        if label == "Positive":
            label = 1
        elif label == "Negative":
            label = 0

        return {
            "patch": self._transform(transforms.ToTensor()(patch.image)),
            "label": label,
            "slide_name": slide_name,
            "center_pixel": center_pixel,
        }


class SlideDataset(WSIDataset):
    def __init__(
        self,
        bags_per_slide: int = 1,
        target: str = "ER",
        tile_size: int = 256,  # TODO: make this useful through patch extraction
        desired_mpp: float = 1.0,  # TODO: make this useful through patch extraction
        bag_size: int = 100,
        metadata_at_magnification: int = 10,
        min_tiles: int = 100,
        dataset: str = "CAT",
        metadata_file_path: str = None,
        datasets_base_dir_path: str = None,
        transform=transforms.Compose([]),
        train: bool = True,  # controls folds
        val_fold: int = None,
        slides_manager: SlidesManager = None,
        **kw: object
    ):
        self._transform = transform
        self._target = BioMarker[target]
        self._train = train
        self._bag_size = bag_size
        super().__init__(
            instances_per_slide=bags_per_slide,
            slides_manager=slides_manager,
            target=target,
            tile_size=tile_size,  # TODO: make this useful through patch extraction
            desired_mpp=desired_mpp,  # TODO: make this useful through patch extraction
            metadata_at_magnification=metadata_at_magnification,
            min_tiles=min_tiles,
            dataset=dataset,
            metadata_file_path=metadata_file_path,
            datasets_base_dir_path=datasets_base_dir_path,
            transform=transform,
            val_fold=val_fold,
            train=train,
            **kw
        )

    @abstractmethod
    def __getitem__(self, item: int):
        pass

    def get_bag(self, item: int):
        slide = self._slides_manager.get_slide(item % self.num_slides)
        
    def get_bag(self, item: int):
        slide = self._slides_manager.get_slide(item % self.num_slides)
        slide_name = slide.slide_context.image_file_name_stem
        self.patch_extractor = self.patch_extractor_constructor(slide=slide)
        patch, center_pixel = self.patch_extractor.extract_patch(patch_validators=[])
        bag_item = transforms.ToTensor()(patch.image)
        bag_shape = (self._bag_size, *bag_item.shape)
        bag = torch.zeros(bag_shape)
        bag[0] = bag_item
        for idx in range(1, self._bag_size):
            patch, center_pixel = self.patch_extractor.extract_patch(
                patch_validators=[]
            )
            bag_item = transforms.ToTensor()(patch.image)
            bag[idx] = bag_item

        label = slide.slide_context.get_biomarker_value(bio_marker=self._target)
        if label == "Positive":
            label = 1
        elif label == "Negative":
            label = 0

        # TODO: add patch locations/coords
        return {"bag": self._transform(bag), "label": label, "slide_name": slide_name}


class SlideRandomDataset(SlideDataset):
    def __init__(
        self,
        bags_per_slide: int = 1,
        target: str = "ER",
        tile_size: int = 256,  # TODO: make this useful through patch extraction
        desired_mpp: float = 1.0,  # TODO: make this useful through patch extraction
        bag_size: int = 100,
        metadata_at_magnification: int = 10,
        min_tiles: int = 100,
        dataset: str = "CAT",
        metadata_file_path: str = None,
        datasets_base_dir_path: str = None,
        transform=transforms.Compose([]),
        train: bool = True,  # controls folds
        val_fold: int = None,
        slides_manager: SlidesManager = None,
        **kw: object
    ):
        self._transform = transform
        self._target = BioMarker[target]
        self._train = train
        self._bag_size = bag_size
        super().__init__(
            bags_per_slide=bags_per_slide,
            slides_manager=slides_manager,
            target=target,
            tile_size=tile_size,  # TODO: make this useful through patch extraction
            desired_mpp=desired_mpp,  # TODO: make this useful through patch extraction
            bag_size=bag_size,
            metadata_at_magnification=metadata_at_magnification,
            min_tiles=min_tiles,
            dataset=dataset,
            metadata_file_path=metadata_file_path,
            datasets_base_dir_path=datasets_base_dir_path,
            transform=transform,
            val_fold=val_fold,
            train=train,
            **kw
        )

    def __getitem__(self, item: int):
        return self.get_bag(item)

    def patch_extractor_constructor(self, slide: Slide = None):
        return RandomPatchExtractor(slide)


class SlideStridedDataset(SlideDataset):
    def __init__(
        self,
        target: str = "ER",
        tile_size: int = 256,  # TODO: make this useful through patch extraction
        desired_mpp: float = 1.0,  # TODO: make this useful through patch extraction
        bag_size: int = 100,
        metadata_at_magnification: int = 10,
        min_tiles: int = 100,
        dataset: str = "CAT",
        metadata_file_path: str = None,
        datasets_base_dir_path: str = None,
        transform=transforms.Compose([]),
        train: bool = True,  # controls folds
        val_fold: int = None,
        slides_manager: SlidesManager = None,
        **kw: object
    ):
        self._transform = transform
        self._target = BioMarker[target]
        self._train = train
        self._bag_size = bag_size
        super().__init__(
            bags_per_slide=1,
            slides_manager=slides_manager,
            target=target,
            tile_size=tile_size,  # TODO: make this useful through patch extraction
            desired_mpp=desired_mpp,  # TODO: make this useful through patch extraction
            bag_size=bag_size,
            metadata_at_magnification=metadata_at_magnification,
            min_tiles=min_tiles,
            dataset=dataset,
            metadata_file_path=metadata_file_path,
            datasets_base_dir_path=datasets_base_dir_path,
            transform=transform,
            val_fold=val_fold,
            train=train,
            **kw
        )

    def __getitem__(self, item: int):
        return self.get_bag(item)

    def patch_extractor_constructor(self, slide: Slide = None):
        return StridedPatchExtractor(slide, self._bag_size)


class SlideGridDataset(SlideDataset):
    def __init__(
        self,
        bags_per_slide: int = 1,
        side_length=3,
        target: str = "ER",
        tile_size: int = 256,  # TODO: make this useful through patch extraction
        desired_mpp: float = 1.0,  # TODO: make this useful through patch extraction
        metadata_at_magnification: int = 10,
        min_tiles: int = 100,
        dataset: str = "CAT",
        metadata_file_path: str = None,
        datasets_base_dir_path: str = None,
        transform=transforms.Compose([]),
        train: bool = True,  # controls folds
        val_fold: int = None,
        slides_manager: SlidesManager = None,
        **kw: object
    ):
        super().__init__(
            bags_per_slide=bags_per_slide,
            slides_manager=slides_manager,
            target=target,
            tile_size=tile_size,  # TODO: make this useful through patch extraction
            desired_mpp=desired_mpp,  # TODO: make this useful through patch extraction
            bag_size=side_length**2,
            metadata_at_magnification=metadata_at_magnification,
            min_tiles=min_tiles,
            dataset=dataset,
            metadata_file_path=metadata_file_path,
            datasets_base_dir_path=datasets_base_dir_path,
            transform=transform,
            val_fold=val_fold,
            train=train,
            **kw
        )
        self._transform = transform
        self._target = BioMarker[target]
        self._train = train
        self._side_length = side_length

    def __getitem__(self, item: int):
        return self.get_bag(item)

    def patch_extractor_constructor(self, slide: Slide = None):
        return GridPatchExtractor(slide, self._side_length)
