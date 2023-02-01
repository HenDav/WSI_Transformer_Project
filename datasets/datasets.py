# python core
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Callable, cast
from enum import Enum
from pathlib import Path
import multiprocessing
import time
import socket

# pandas
import pandas

# numpy
import numpy
import torch

# torch
from torch.utils.data import Dataset
from torchvision import transforms

# gipmed
from wsi_core.metadata import MetadataBase
from wsi_core import constants
from wsi_core.base import SeedableObject
from wsi_core.wsi import SlideContext, Tile, Slide, Patch, PatchExtractor, RandomPatchExtractor, ProximatePatchExtractor, BioMarker
from wsi_core.parallel_processing import TaskParallelProcessor, OnlineParallelProcessor, InfiniteOnlineParallelProcessor, FiniteOnlineParallelProcessor, GetItemPolicy


# =================================================
# SlidesManager Class
# =================================================
class SlidesManager(SeedableObject, MetadataBase):
    def __init__(
            self,
            datasets_base_dir_path: Path,
            tile_size: int,
            desired_mpp: float,
            metadata_at_magnification: int,
            metadata_file_path: Path,
            row_predicate: Callable, #[[pandas.DataFrame, ...], pandas.Index] somehow causes a bug, I have no idea why
            **predicate_args):
        if "folds" in predicate_args.keys():
            assert self.datasets_have_compatible_folds(predicate_args["datasets"]), "datasets do not have compatible folds"
        self._desired_mpp = desired_mpp
        self._metadata_file_path = metadata_file_path
        self._slides = []
        self._current_slides = []
        self._slides_with_interior = []
        # self._tile_to_slide_dict = self._create_tile_to_slide_dict()
        MetadataBase.__init__(self, datasets_base_dir_path=datasets_base_dir_path, tile_size=tile_size, metadata_at_magnification=metadata_at_magnification, desired_mpp=desired_mpp)
        SeedableObject.__init__(self)
        self._current_slides = self._create_slides()
        self._current_df = self._df.iloc[row_predicate(self._df, **predicate_args)]
        # self.start()
        # self.join()

    def __len__(self) -> int:
        return len(self._current_df)
    
    def _create_slides(self) -> List[Slide]:
        slides = []
        for row_index in range(self._df.shape[0]):
            slide_context = SlideContext(row_index=row_index, metadata=self._df, dataset_paths=self._dataset_paths, desired_mpp=self._desired_mpp, tile_size=self._tile_size)
            slide = Slide(slide_context=slide_context)
            slides.append(slide)

        # self._df = self._update_metadata()
        self._file_name_to_slide = self._create_file_name_to_slide_dict()
        # self.filter_folds(folds=None)

        return slides
    
    @staticmethod
    def datasets_have_compatible_folds(datasets: List[str]) -> bool:
        assert len(datasets)>0, "datasets list should not be empty"
        compatible_folds = True
        datasets_folds = [constants.folds_for_datasets[dataset] for dataset in datasets]
        for dataset_folds in datasets_folds:
            compatible_folds = compatible_folds and datasets_folds[0] == dataset_folds
        return compatible_folds

    @property
    def metadata(self) -> pandas.DataFrame:
        return self._df

    @property
    def slides_count(self) -> int:
        return self._current_df.shape[0]

    # def get_slide_by_tile(self, tile: Tile) -> Slide:
    #     return self._tile_to_slide_dict[tile]

    def get_slides_ratio(self, ratio: float) -> List[Slide]:
        modulo = int(1 / ratio)
        return self.get_slides_modulo(modulo=modulo)

    def get_slides_modulo(self, modulo: int) -> List[Slide]:
        return self._slides[::modulo]

    def filter_folds(self, folds: Optional[List[int]]):
        if folds is not None:
            self._current_df = self._df[self._df[constants.fold_column_name].isin(folds)]
        else:
            self._current_df = self._df

        self._current_slides = self._get_slides()
        # self._slides_with_interior = self._get_slides_with_interior_tiles()

    def get_slide(self, index: int) -> Slide:
        return self._current_slides[index]

    def get_random_slide(self) -> Slide:
        index = self._rng.integers(low=0, high=self._current_df.shape[0])
        return self.get_slide(index=index)

    def get_slide_with_interior(self, index: int) -> Slide:
        return self._slides_with_interior[index]

    def get_random_slide_with_interior(self) -> Slide:
        index = self._rng.integers(low=0, high=len(self._slides_with_interior))
        return self.get_slide_with_interior(index=index)

    def _add_shared_objects(self, namespace: multiprocessing.managers.Namespace):
        namespace.metadata = self._df

    def _load_metadata(self) -> pandas.DataFrame:
        return pandas.read_csv(filepath_or_buffer=self._metadata_file_path)

    # def _generate_exempt_from_pickle(self) -> List[str]:
    #     # return []
    #     exempt_from_pickle = super()._generate_exempt_from_pickle()
    #     # exempt_from_pickle.append('_df')
    #     # exempt_from_pickle.append('_current_df')
    #     return exempt_from_pickle

    # def _post_join(self):
    #     self._slides = self._collect_slides()
    #
    #     # total_slides = 0
    #     # for slide in self._slides:
    #     #     total_slides = total_slides + slide.tiles_count
    #
    #     self._df = self._update_metadata()
    #     # # print(f'TASKS LEN {len(self._tasks)}')
    #     # # print(f'SLIDES LEN {len(self._slides)}')
    #     self._file_name_to_slide = self._create_file_name_to_slide_dict()
    #     self.filter_folds(folds=None)

    def _update_metadata(self) -> pandas.DataFrame:
        image_file_names = [slide.slide_context.image_file_name for slide in self._slides]
        return self._df[self._df[constants.file_column_name].isin(image_file_names)]

    # def _collect_slides(self) -> List[Slide]:
    #     completed_tasks = cast(List[SlidesManagerTask], self._completed_tasks)
    #     return [task.slide for task in completed_tasks if task.slide is not None]

        # slides = []
        # for i, task in enumerate(self._completed_tasks):
        #     task = cast(SlidesManagerTask, task)
        #     if task.slide is not None:
        #         slides.append(task.slide)
        #
        # return slides

    # def _generate_tasks(self) -> List[ParallelProcessorTask]:
    #     tasks = []
    #
    #     combinations = list(itertools.product(*[
    #         [*range(self._df.shape[0])],
    #         [self._dataset_paths],
    #         [self._metadata_at_magnification],
    #         [self._tile_size]]))
    #
    #     for combination in combinations:
    #         tasks.append(SlidesManagerTask(
    #             row_index=combination[0],
    #             dataset_paths=combination[1],
    #             desired_magnification=combination[2],
    #             tile_size=combination[3]))
    #
    #     return tasks

    def _create_file_name_to_slide_dict(self) -> Dict[str, Slide]:
        file_name_to_slide = {}
        # print(f'slides len {len(self._slides)}')
        for slide in self._slides:
            # print(slide.slide_context.image_file_name)
            file_name_to_slide[slide.slide_context.image_file_name] = slide

        return file_name_to_slide

    def _create_tile_to_slide_dict(self) -> Dict[Tile, Slide]:
        tile_to_slide = {}
        for slide in self._slides:
            for tile in slide.tiles:
                tile_to_slide[tile] = slide

        return tile_to_slide

    def _get_slides(self) -> List[Slide]:
        # print(self._current_df[constants.file_column_name])
        # print(self._file_name_to_slide)
        return [self._file_name_to_slide[x] for x in self._current_df[constants.file_column_name]]

    def _get_slides_with_interior_tiles(self) -> List[Slide]:
        slides_with_interior_tiles = []
        for slide in self._slides:
            if slide.interior_tiles_count > 0:
                slides_with_interior_tiles.append(slide)

        return slides_with_interior_tiles

    
# =================================================
# TupletsDataset Class
# =================================================
class WSIDataset(ABC, Dataset, SeedableObject):
    def __init__(self, 
                 instances_per_slide: int = 10, 
                 target: str = 'ER', 
                 tile_size: int = 256,
                 desired_mpp: float = 1.0,
                 metadata_at_magnification: int = 10,
                 min_tiles: int = 100,
                 dataset: str = 'CAT',
                 metadata_file_path: str = None,
                 datasets_base_dir_path: str = None,
                 transform = transforms.Compose([transforms.ToTensor()]),
                 val_fold: int = None,
                 train: bool = True,
                 slides_manager: SlidesManager = None, 
                 **kw: object):
        super().__init__(**kw)
        if not slides_manager:
            if not datasets_base_dir_path:
                if socket.gethostname()=='gipdeep10':
                    datasets_base_dir_path = constants.data_root_gipdeep10
                else:
                    datasets_base_dir_path = constants.data_root_netapp
            assert len(datasets_base_dir_path)>0, "Problem with datasets_base_dir_path path"
            if not metadata_file_path:
                metadata_file_path = constants.main_metadata_csv
            datasets = constants.get_dataset_ids(dataset)
            current_folds = constants.folds_for_datasets[next(iter(datasets))]
            if val_fold != None and train:
                current_folds.remove(val_fold)
            elif val_fold != None and not train:
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
                folds=current_folds
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
    def default_predicate(df: pandas.DataFrame = None, 
                          min_tiles: int = 100, 
                          datasets: List[str] = ['CARMEL1'],
                          folds: List[int] = constants.folds_for_datasets['CARMEL1']) -> pandas.Index:
        fold_indecies = df.index[df[constants.fold_column_name].isin(folds)]
        print(fold_indecies)
        dataset_indecies = df.index[df[constants.dataset_id_column_name].isin(datasets)]
        print(dataset_indecies)
        min_tiles_indecies = df.index[df[constants.legitimate_tiles_column_name] > min_tiles]
        print(min_tiles_indecies)
        return min_tiles_indecies.intersection(dataset_indecies).intersection(fold_indecies)

class PatchSupervisedDataset(WSIDataset):
    
    def __init__(self, 
                 patches_per_slide: int = 10, 
                 target: str = 'ER', 
                 tile_size: int = 256, #TODO: make this useful through patch extraction
                 desired_mpp: float = 1.0, #TODO: make this useful through patch extraction
                 metadata_at_magnification: int = 10,
                 min_tiles: int = 100,
                 dataset: str = 'CAT',
                 metadata_file_path: str = None,
                 datasets_base_dir_path: str = None,
                 transform = transforms.Compose([transforms.ToTensor()]),
                 val_fold: int = None,
                 train: bool = True, # TODO: maybe remove this?
                 slides_manager: SlidesManager = None, 
                 **kw: object):
        self._transform = transform
        self._target = BioMarker[target]
        self._train = train
        super().__init__(instances_per_slide=patches_per_slide,
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
                         **kw)

    def __getitem__(self, item: int):
        slide = self._slides_manager.get_slide(item % self.num_slides)
        
        patch_extractor = RandomPatchExtractor(slide=slide)
        patch = patch_extractor.extract_patch(patch_validators=[])
        label = slide.slide_context.get_biomarker_value(bio_marker=self._target)
        if label == "Positive":
            label = 1
        elif label == "Negative":
            label = 0

        return self._transform(patch.image), label
        
    
class SlideSupervisedDataset(WSIDataset):
    
    def __init__(self, 
                 bags_per_slide: int = 1,
                 target: str = 'ER', 
                 tile_size: int = 256, #TODO: make this useful through patch extraction
                 desired_mpp: float = 1.0, #TODO: make this useful through patch extraction
                 bag_size: int = 100,
                 metadata_at_magnification: int = 10,
                 min_tiles: int = 100,
                 dataset: str = 'CAT',
                 metadata_file_path: str = None,
                 datasets_base_dir_path: str = None,
                 transform = transforms.Compose([transforms.ToTensor()]),
                 val_fold: int = None,
                 train: bool = True,
                 slides_manager: SlidesManager = None, 
                 **kw: object):
        self._transform = transform
        self._target = BioMarker[target]
        self._train = train
        self._bag_size = bag_size
        assert self._bag_size<=min_tiles, "bag size needs to be smaller than min_tiles"
        assert train or bags_per_slide==1, "since validation/test behaviour is constant, more than 1 bag per slide on validation is meaningless"
        super().__init__(instances_per_slide=bags_per_slide,
                         slides_manager=slides_manager,
                         target=target, 
                         tile_size=tile_size, #TODO: make this useful through patch extraction
                         desired_mpp=desired_mpp, #TODO: make this useful through patch extraction
                         metadata_at_magnification=metadata_at_magnification,
                         min_tiles=min_tiles,
                         dataset=dataset,
                         metadata_file_path=metadata_file_path,
                         datasets_base_dir_path=datasets_base_dir_path,
                         transform=transform,
                         val_fold=val_fold,
                         train=train,
                         **kw)
        self.patch_extractor_constructor = self.patch_extractor_constructor_train if train else self.patch_extractor_constructor_test

    def __getitem__(self, item: int):

        slide = self._slides_manager.get_slide(item % self.num_slides)
        
        patch_extractor = self.patch_extractor_constructor(slide=slide)
        patch = patch_extractor.extract_patch(patch_validators=[])
        bag_item = self._transform(patch.image)
        bag = torch.zeros((self._bag_size, *bag_item.size()))
        bag[0] = bag_item
        for idx in range(1, self._bag_size):
            patch = patch_extractor.extract_patch(patch_validators=[])
            bag_item = self._transform(patch.image)
            bag[idx] = bag_item
            
        label = slide.slide_context.get_biomarker_value(bio_marker=self._target)
        if label == "Positive":
            label = 1
        elif label == "Negative":
            label = 0

        return bag, label
    
    def patch_extractor_constructor_train(slide: Slide = None):
        RandomPatchExtractor(slide)
        
    def patch_extractor_constructor_test(slide: Slide = None):
        StridedPatchExtractor(slide, self._bag_size)
        

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
