import socket
from abc import ABC, abstractmethod
from typing import List, Optional, Dict
from datetime import datetime

import pandas
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import to_tensor

from .slides_manager import SlidesManager, default_predicate
from ..core import constants
from ..core.wsi import (
    GridPatchExtractor,
    MultiGridPatchExtractor,
    RandomPatchExtractor,
    SinglePatchExtractor,
    Slide,
    StridedPatchExtractor,
    Patch
)

class WSIDataset(ABC, Dataset):
    def __init__(
        self,
        instances_per_slide: int = 10,
        target: str = "er_status",
        secondary_target: str = None,
        tile_size: int = 256,
        desired_mpp: float = 1.0,
        metadata_at_magnification: int = 10,
        min_tiles: int = 100,
        metadata_file_path: Optional[str] = None,
        datasets_base_dir_path: Optional[str] = None,
        transform=transforms.Compose([]),
        datasets_folds: Dict = {"CAT": [2,3,4,5]},
        slides_manager: SlidesManager = None,
        **kw: object,
    ):
        if kw:
            print(kw)
        super().__init__(**kw)
        self._target = target
        self._secondary_target = secondary_target
        if not slides_manager:
            if datasets_base_dir_path:
                assert (
                    len(datasets_base_dir_path) > 0
                ), "Problem with datasets_base_dir_path path"
                print(f"Overriding datasets_base_dir_path: {datasets_base_dir_path}")
            else:
                if socket.gethostname() == "gipdeep10":
                    datasets_base_dir_path = constants.data_root_gipdeep10
                    print("Using gipdeep10 data.")
                else:
                    datasets_base_dir_path = constants.data_root_netapp
            if not metadata_file_path:
                metadata_file_path = constants.main_metadata_csv
            datasets_folds = constants.get_datasets_folds(datasets_folds)
            self.datasets_keys = list(datasets_folds.keys())

            slides_manager = SlidesManager(
                datasets_base_dir_path=datasets_base_dir_path,
                desired_mpp=desired_mpp,
                tile_size=tile_size,
                metadata_at_magnification=metadata_at_magnification,
                metadata_file_path=metadata_file_path,
                row_predicate=default_predicate,
                min_tiles=min_tiles,
                datasets_folds=datasets_folds,
                target=target,
                secondary_target=secondary_target
            )
        self._slides_manager = slides_manager
        self.num_slides = len(slides_manager)
        self._instances_per_slide = instances_per_slide
        self._dataset_size = instances_per_slide * self.num_slides

    def __len__(self):
        return self._dataset_size


class RandomPatchDataset(WSIDataset):
    def __init__(
        self,
        patches_per_slide: int = 10,
        target: str = "er_status",
        secondary_target: str = None,
        tile_size: int = 256,  # TODO: make this useful through patch extraction
        desired_mpp: float = 1.0,  # TODO: make this useful through patch extraction
        metadata_at_magnification: int = 10,
        min_tiles: int = 100,
        metadata_file_path: str = None,
        datasets_base_dir_path: str = None,
        transform=transforms.Compose([]),
        datasets_folds: Dict = {"CAT": [2,3,4,5]},
        slides_manager: SlidesManager = None,
        **kw: object,
    ):
        super().__init__(
            instances_per_slide=patches_per_slide,
            slides_manager=slides_manager,
            target=target,
            secondary_target=secondary_target,
            tile_size=tile_size,
            desired_mpp=desired_mpp,
            metadata_at_magnification=metadata_at_magnification,
            min_tiles=min_tiles,
            datasets_folds=datasets_folds,
            metadata_file_path=metadata_file_path,
            datasets_base_dir_path=datasets_base_dir_path,
            transform=transform,
            **kw,
        )
        self._transform = transform

    def __getitem__(self, item: int):
        slide = self._slides_manager.get_slide(item // self._instances_per_slide)

        slide_name = slide.slide_context.image_file_name
        dataset_id = self.datasets_keys.index(slide.slide_context.dataset_id)
        patch_extractor = RandomPatchExtractor(slide=slide)
        patch, center_pixel = patch_extractor.extract_patch(patch_validators=[])
        label = slide.slide_context.get_biomarker_value(bio_marker=self._target)

        item =  {
            "patch": self._transform(patch.image),
            "label": label,
            "slide_name": slide_name,
            "dataset_id": dataset_id,
            "center_pixel": center_pixel,
            }

        if self._secondary_target:
            secondary_label = slide.slide_context.get_biomarker_value(bio_marker=self._secondary_target)
            item["secondary_label"] = secondary_label

        return item

class SerialPatchDataset(WSIDataset):
    def __init__(
        self,
        target: str = "er_status",
        secondary_target: str = None,
        tile_size: int = 256,  # TODO: make this useful through patch extraction
        desired_mpp: float = 1.0,  # TODO: make this useful through patch extraction
        metadata_at_magnification: int = 10,
        min_tiles: int = 100,
        metadata_file_path: str = None,
        datasets_base_dir_path: str = None,
        transform=transforms.Compose([]),
        datasets_folds: Dict = {"CAT": [2,3,4,5]},
        slides_manager: SlidesManager = None,
        **kw: object,
    ):
        super().__init__(
            instances_per_slide=1,  # will not be used
            slides_manager=slides_manager,
            target=target,
            secondary_target=secondary_target,
            tile_size=tile_size,
            desired_mpp=desired_mpp,
            metadata_at_magnification=metadata_at_magnification,
            min_tiles=min_tiles,
            datasets_folds=datasets_folds,
            metadata_file_path=metadata_file_path,
            datasets_base_dir_path=datasets_base_dir_path,
            transform=transform,
            **kw,
        )
        self._dataset_size = self._slides_manager.tiles_count
        self._transform = transform
        self._n_slides = 0
        self._n_slide_patches = 0

    def __getitem__(self, item: int):
        self._tile = self._slides_manager.get_tile(item)
        center_pixel = self._tile.center_pixel
        patch = Patch(
            slide_context=self._tile.slide_context, center_pixel=center_pixel
        )
        slide_name = self._tile.slide_context.image_file_name
        dataset_id = self._tile.slide_context.dataset_id
        label = self._tile.slide_context.get_biomarker_value(bio_marker=self._target)
        
        patch_img = patch.image
        patch_img = self._transform(patch_img)
        item = {
            "patch": patch_img,
            "label": label,
            "slide_name": slide_name,
            "dataset_id": dataset_id,
            "center_pixel": center_pixel,
            }
        
        if self._secondary_target:
            secondary_label = self._tile.slide_context.get_biomarker_value(bio_marker=self._secondary_target)
            item["secondary_label"] = secondary_label
        
        return item


class SlideDataset(WSIDataset):
    def __init__(
        self,
        bags_per_slide: int = 1,
        target: str = "er_status",
        secondary_target: str = None,
        tile_size: int = 256,  # TODO: make this useful through patch extraction
        desired_mpp: float = 1.0,  # TODO: make this useful through patch extraction
        bag_size: int = 100,
        metadata_at_magnification: int = 10,
        min_tiles: int = 100,
        datasets_folds: Dict = {"CAT": [2,3,4,5]},
        metadata_file_path: str = None,
        datasets_base_dir_path: str = None,
        transform=transforms.Compose([]),
        slides_manager: SlidesManager = None,
        **kw: object,
    ):
        self._transform = transform
        self._bag_size = bag_size
        super().__init__(
            instances_per_slide=bags_per_slide,
            slides_manager=slides_manager,
            target=target,
            secondary_target=secondary_target,
            tile_size=tile_size,  # TODO: make this useful through patch extraction
            desired_mpp=desired_mpp,  # TODO: make this useful through patch extraction
            metadata_at_magnification=metadata_at_magnification,
            min_tiles=min_tiles,
            datasets_folds=datasets_folds,
            metadata_file_path=metadata_file_path,
            datasets_base_dir_path=datasets_base_dir_path,
            transform=transform,
            **kw,
        )

    @abstractmethod
    def __getitem__(self, item: int):
        pass

    def get_bag(self, item: int):
        slide = self._slides_manager.get_slide(item // self._instances_per_slide)
        slide_name = slide.slide_context.image_file_name
        dataset_id = self.datasets_keys.index(slide.slide_context.dataset_id)
        self.patch_extractor = self.patch_extractor_constructor(slide=slide)
        patch, center_pixel = self.patch_extractor.extract_patch(patch_validators=[])
        bag_item = to_tensor(patch.image)
        bag_shape = (self._bag_size, *bag_item.shape)
        bag = torch.zeros(bag_shape)
        center_pixels = []
        bag[0] = self._transform(patch.image)
        center_pixels.append(center_pixel)
        for idx in range(1, self._bag_size):
            patch, center_pixel = self.patch_extractor.extract_patch(
                patch_validators=[]
            )
            center_pixels.append(center_pixel)
            bag_item = self._transform(patch.image)

            bag[idx] = bag_item

        label = slide.slide_context.get_biomarker_value(bio_marker=self._target)
        label = torch.tensor(label)
        label = label.expand((self._bag_size, *label.shape)).clone()
        dataset_id = torch.tensor(dataset_id)

        # TODO: add patch locations/coords
        item = {
            "bag": bag, 
            "label": label,
            "slide_name": slide_name, 
            "dataset_id": dataset_id, 
            "center_pixels": center_pixels
            }
        if self._secondary_target:
            secondary_label = slide.slide_context.get_biomarker_value(bio_marker=self._secondary_target)
            secondary_label = torch.tensor(secondary_label).expand(self._bag_size).clone()
            item["secondary_label"] = secondary_label
        return item


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
        datasets_folds: Dict = {"CAT": [2,3,4,5]},
        metadata_file_path: str = None,
        datasets_base_dir_path: str = None,
        transform=transforms.Compose([]),
        slides_manager: SlidesManager = None,
        **kw: object,
    ):
        self._transform = transform
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
            datasets_folds=datasets_folds,
            metadata_file_path=metadata_file_path,
            datasets_base_dir_path=datasets_base_dir_path,
            transform=transform,
            **kw,
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
        datasets_folds: Dict = {"CAT": [2,3,4,5]},
        metadata_file_path: str = None,
        datasets_base_dir_path: str = None,
        transform=transforms.Compose([]),
        slides_manager: SlidesManager = None,
        **kw: object,
    ):
        self._transform = transform
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
            datasets_folds=datasets_folds,
            metadata_file_path=metadata_file_path,
            datasets_base_dir_path=datasets_base_dir_path,
            transform=transform,
            **kw,
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
        datasets_folds: Dict = {"CAT": [2,3,4,5]},
        metadata_file_path: str = None,
        datasets_base_dir_path: str = None,
        transform=transforms.Compose([]),
        slides_manager: SlidesManager = None,
        **kw: object,
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
            datasets_folds=datasets_folds,
            metadata_file_path=metadata_file_path,
            datasets_base_dir_path=datasets_base_dir_path,
            transform=transform,
            **kw,
        )
        self._transform = transform
        self._side_length = side_length

    def __getitem__(self, item: int):
        return self.get_bag(item)

    def patch_extractor_constructor(self, slide: Slide = None):
        return GridPatchExtractor(slide, self._side_length)

class SlideMultiGridDataset(SlideDataset):
    """
    A dataset which instead of sampling a grid samples multiple grids in a bag, while
    resulting in an overall square grid.
    """
    def __init__(
        self,
        bags_per_slide: int = 1,
        side_length: int=5,
        num_grids: int=4,
        target: str = "ER",
        tile_size: int = 256,  # TODO: make this useful through patch extraction
        desired_mpp: float = 1.0,  # TODO: make this useful through patch extraction
        metadata_at_magnification: int = 10,
        min_tiles: int = 100,
        datasets_folds: Dict = {"CAT": [2,3,4,5]},
        metadata_file_path: str = None,
        datasets_base_dir_path: str = None,
        transform=transforms.Compose([]),
        slides_manager: SlidesManager = None,
        **kw: object,
    ):
        # Assert that num_grids is a square of an int.
        assert num_grids == int(num_grids**0.5)**2, f"num_grids is {num_grids}. It must be a square of an int."
        super().__init__(
            bags_per_slide=bags_per_slide,
            slides_manager=slides_manager,
            target=target,
            tile_size=tile_size,  # TODO: make this useful through patch extraction
            desired_mpp=desired_mpp,  # TODO: make this useful through patch extraction
            bag_size=side_length**2,
            metadata_at_magnification=metadata_at_magnification,
            min_tiles=min_tiles,
            datasets_folds=datasets_folds,
            metadata_file_path=metadata_file_path,
            datasets_base_dir_path=datasets_base_dir_path,
            transform=transform,
            **kw,
        )
        self._transform = transform
        self._side_length = side_length
        self._num_grids = num_grids

    def __getitem__(self, item: int):
        return self.get_bag(item)

    def patch_extractor_constructor(self, slide: Slide = None):
        return MultiGridPatchExtractor(slide, self._side_length, self._num_grids)