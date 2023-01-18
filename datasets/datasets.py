# python core
from abc import ABC, abstractmethod
from typing import List, Optional, Dict
from enum import Enum
from pathlib import Path
import multiprocessing
import time

# pandas
import pandas

# numpy
import numpy
import torch

# torch
from torch.utils.data import Dataset

# gipmed
from wsi_core.metadata import SlidesManager
from wsi_core import constants
from wsi_core.base import SeedableObject
from wsi_core.wsi import SlideContext, Tile, Slide, Patch, PatchExtractor, RandomPatchExtractor, ProximatePatchExtractor, BioMarker
from wsi_core.parallel_processing import TaskParallelProcessor, OnlineParallelProcessor, InfiniteOnlineParallelProcessor, FiniteOnlineParallelProcessor, GetItemPolicy


# =================================================
# TupletsDataset Class
# =================================================
class WSIDataset(ABC, Dataset, SeedableObject):
    def __init__(self, slides_manager: SlidesManager, dataset_size: int, **kw: object):
        super().__init__(**kw)
        self._dataset_size = dataset_size
        self._slides_manager = slides_manager

    def __len__(self):
        return self._dataset_size

    # @abstractmethod
    # def __getitem__(self, item: int) -> object:
    #     pass

    def set_folds(self, folds: List[int]):
        self._slides_manager.filter_folds(folds=folds)


class WSIDatasetTrain(WSIDataset):
    
    def __init__(self, 
                 patches_per_slide: int, 
                 target: BioMarker, 
                 tile_size: int, #TODO: make this useful through patch extraction
                 desired_mpp: float = 1.0, #TODO: make this useful through patch extraction
                 metadata_at_magnification: int = 10,
                 min_tiles: int = 100,
                 metadata_file_path: Path = None,
                 datasets_base_dir_path: Path = None,
                 slides_manager: SlidesManager = None, 
                 **kw: object):
        if not slides_manager:
            slides_manager = SlidesManager(
                datasets_base_dir_path=datasets_base_dir_path,
                desired_mpp=desired_mpp,
                tile_size=tile_size,
                metadata_at_magnification=metadata_at_magnification,
                metadata_file_path=metadata_file_path,
                row_predicate=self.default_predicate,
                min_tiles=min_tiles
            )
        self.num_slides = len(slides_manager)
        self._target = target
        super().__init__(slides_manager=slides_manager, dataset_size=self.num_slides*patches_per_slide, **kw)

    def __getitem__(self, item: int):

        # start_time = time.time_ns()
        slide = self._slides_manager.get_slide(item % self.num_slides)
        # end_time = time.time_ns()
        # delta = (start_time - end_time) / (10 ** 9)
        # print(f'Line 1: {delta}')


        # start_time = time.time_ns()
        patch_extractor = RandomPatchExtractor(slide=slide)
        # end_time = time.time_ns()
        # delta = (start_time - end_time) / (10 ** 9)
        # print(f'Line 2: {delta}')

        # start_time = time.time_ns()
        patch = patch_extractor.extract_patch(patch_validators=[])
        label = slide.slide_context.get_biomarker_value(bio_marker=self._target)

        # end_time = time.time_ns()
        # delta = (start_time - end_time) / (10 ** 9)
        # print(f'Line 3: {delta}')

        # return torch.Tensor()
        return patch.image, label

        # if WSIDatasetTest.image is None:
        # WSIDatasetTest.image = patch.image
        #
        # return WSIDatasetTest.image
        # return patch.image
    
    @staticmethod
    def default_predicate(df: pandas.DataFrame = None, min_tiles: int = 100) -> pandas.Index:
        return df.index[df[constants.legitimate_tiles_column_name] > min_tiles]



# =================================================
# WSIMultiProcessingDataset Class
# =================================================
# class WSIMultiProcessingDataset(OnlineParallelProcessor, WSIDataset):
#     def __init__(self, name: str, output_dir_path: Path, num_workers: int, items_queue_maxsize: int, items_buffer_size: int, slides_manager: SlidesManager, dataset_size: int):
#         super().__init__(name=name, output_dir_path=output_dir_path, num_workers=num_workers, items_queue_maxsize=items_queue_maxsize, items_buffer_size=items_buffer_size, slides_manager=slides_manager, dataset_size=dataset_size)
#         self.start()
#
#     @abstractmethod
#     def _generate_item(self) -> object:
#         pass
#
#     def __getitem__(self, index):
#         return self.get_item(index=index)


# =================================================
# SingleTargetTrainingDataset Class
# =================================================
class SingleTargetTrainingDataset(WSIDataset, InfiniteOnlineParallelProcessor):
    def __init__(self, name: str, output_dir_path: Path, num_workers: int, items_queue_maxsize: int, items_buffer_size: int, slides_manager: SlidesManager, dataset_size: int, target: BioMarker):
        super().__init__(name=name, output_dir_path=output_dir_path, num_workers=num_workers, items_queue_maxsize=items_queue_maxsize, items_buffer_size=items_buffer_size, slides_manager=slides_manager, dataset_size=dataset_size, get_item_policy=GetItemPolicy.TryReplace)
        self._target = target

    def _generate_item(self, item_id: Optional[int]) -> object:
        slide = self._slides_manager.get_random_slide()
        patch_extractor = RandomPatchExtractor(slide=slide)
        patch = patch_extractor.extract_patch(patch_validators=[])
        label = slide.slide_context.get_biomarker_value(bio_marker=self._target)
        return patch, label


# =================================================
# SingleTargetValidationDataset Class
# =================================================
class SingleTargetValidationDataset(WSIDataset, FiniteOnlineParallelProcessor):
    def __init__(
            self,
            slides_manager: SlidesManager,
            slides_ratio: int,
            tiles_ratio: int,
            bio_marker: BioMarker):
        self._slides_ratio = slides_ratio
        self._tiles_ratio = tiles_ratio
        self._bio_marker = bio_marker
        self._tiles = self._get_tiles()
        super().__init__(slides_manager=slides_manager, dataset_size=len(self._tiles))

    def _generate_item(self, item_id: Optional[int]) -> object:
        patch_images = []
        tile = self._tiles[item_id]
        slide = self._slides_manager.get_slide_by_tile(tile=tile)
        patch_images.append(tile.image)
        label = slide.slide_context.get_biomarker_value(bio_marker=self._bio_marker)
        images_tuplet = numpy.transpose(numpy.stack(patch_images), (0, 3, 1, 2))

        return {
            'input': images_tuplet,
            'label': label,
            'slide_id': slide.slide_context.row_index
        }

    def _get_tiles(self) -> List[Tile]:
        tiles = []
        for slide in self._slides_manager.get_slides_ratio(ratio=self._slides_ratio):
            for tile in slide.get_tiles_ratio(ratio=self._tiles_ratio):
                tiles.append(tile)

        return tiles


# =================================================
# SSLDataset Class
# =================================================
class SSLDataset(WSIDataset, InfiniteOnlineParallelProcessor):
    _white_ratio_threshold = 0.5
    _white_intensity_threshold = 170

    def __init__(
            self,
            name: str,
            output_dir_path: Path,
            num_workers: int,
            items_queue_maxsize: int,
            items_buffer_size: int,
            slides_manager: SlidesManager,
            dataset_size: int,
            inner_radius_mm: float,
            negative_examples_count: int):
        super().__init__(
            name=name,
            output_dir_path=output_dir_path,
            num_workers=num_workers,
            items_queue_maxsize=items_queue_maxsize,
            items_buffer_size=items_buffer_size,
            get_item_policy=GetItemPolicy.TryReplace,
            dataset_size=dataset_size,
            slides_manager=slides_manager)
        self._inner_radius_mm = inner_radius_mm
        self._negative_examples_count = negative_examples_count

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['_workers']
        del d['_manager']
        del d['_namespace']
        del d['_slides_manager']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def _generate_item(self, item_id: Optional[int], namespace: multiprocessing.managers.Namespace) -> object:
        return 5
        # while True:
        #     patch_images = []
        #
        #     slide = namespace.slides_manager.get_random_slide_with_interior()
        #     patch_extractor = RandomPatchExtractor(slide=slide)
        #     anchor_patch = patch_extractor.extract_patch(patch_validators=[SSLDataset._validate_histogram])
        #     if anchor_patch is None:
        #         continue
        #
        #     patch_images.append(numpy.array(anchor_patch.image))
        #
        #     patch_extractor = ProximatePatchExtractor(slide=slide, reference_patch=anchor_patch, inner_radius_mm=self._inner_radius_mm)
        #     positive_patch = patch_extractor.extract_patch(patch_validators=[])
        #     if positive_patch is None:
        #         continue
        #
        #     patch_images.append(numpy.array(positive_patch.image))
        #
        #     # for i in range(negative_examples_count):
        #     #     pass
        #
        #     images_tuplet = numpy.transpose(numpy.stack(patch_images), (0, 3, 1, 2))
        #     return images_tuplet

    def _add_shared_objects(self, namespace: multiprocessing.managers.Namespace):
        # pass
        namespace.slides_manager = self._slides_manager

    @staticmethod
    def _validate_histogram(patch: Patch) -> bool:
        patch_grayscale = patch.image.convert('L')
        hist, _ = numpy.histogram(a=patch_grayscale, bins=patch.slide_context.tile_size)
        white_ratio = numpy.sum(hist[SSLDataset._white_intensity_threshold:]) / (patch.slide_context.tile_size * patch.slide_context.tile_size)
        if white_ratio > SSLDataset._white_ratio_threshold:
            return False
        return True
