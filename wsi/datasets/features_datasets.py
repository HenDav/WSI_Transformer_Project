from pathlib import Path
from os.path import join
from typing import Optional, Dict

import h5py
import numpy as np
import torch

from wsi.datasets.datasets import SlideGridDataset, SlideMultiGridDataset, SlideRandomDataset, SlideStridedDataset


class SlideGridFeaturesDataset(SlideGridDataset):
    def __init__(
        self,
        features_dir: str,
        bags_per_slide: int = 1,
        side_length=8,
        target: str = "er_status",
        min_tiles: int = 100,
        datasets_folds: Dict = {'CAT':[2,3,4,5]},
        metadata_file_path: str = None,
        **kw: object,
    ):
        super().__init__(
            bags_per_slide=bags_per_slide,
            side_length=side_length,
            target=target,
            min_tiles=min_tiles,
            datasets_folds=datasets_folds,
            metadata_file_path=metadata_file_path,
            **kw,
        )
        self.features_dir = features_dir
        self._bag_size = (side_length)**2

    def get_bag(self, item: int):
        slide = self._slides_manager.get_slide(item // self._instances_per_slide)
        slide_name = slide.slide_context.image_file_name
        h5_file_name = f"{slide_name}_features.h5"

        with h5py.File(Path(join(self.features_dir, h5_file_name)), "r") as h5_file:
            slide_features = h5_file["features"][:]
            slide_coords = h5_file["coords"][:]

        patch_extractor = self.patch_extractor_constructor(slide=slide)
        bag_coords = patch_extractor._pixels_to_extract.astype("int32")

        features_dim = slide_features.shape[1]
        bag = np.zeros((self._bag_size, features_dim), dtype="float32")

        slide_coords_str = np.array([np.array2string(x) for x in slide_coords.astype("int32")])
        bag_coords_str = np.array([np.array2string(x) for x in bag_coords.astype("int32")])

        existing_features_indices_in_slide = np.where(
            np.isin(slide_coords_str, bag_coords_str)
        )[0]
        existing_features_indices_in_bag = np.where(
            np.isin(bag_coords_str, slide_coords_str)
        )[0]

        assert len(existing_features_indices_in_slide) == len(
            existing_features_indices_in_bag
        ), f"{len(existing_features_indices_in_slide)=} != {len(existing_features_indices_in_bag)=}"

        bag[existing_features_indices_in_bag] = slide_features[
            existing_features_indices_in_slide
        ]

        bag, bag_coords = torch.from_numpy(bag), torch.from_numpy(bag_coords)

        label = slide.slide_context.get_biomarker_value(bio_marker=self._target)
        if label == "Positive":
            label = 1
        elif label == "Negative":
            label = 0
        label = torch.tensor(label)

        return {
            "features": bag,
            "coords": bag_coords,
            "label": label,
            "slide_name": slide_name,
        }
    
class SlideMultiGridFeaturesDataset(SlideMultiGridDataset):
    def __init__(
        self,
        features_dir: str,
        bags_per_slide: int = 1,
        side_length=4,
        num_grids=4,
        target: str = "er_status",
        min_tiles: int = 100,
        datasets_folds: Dict = {'CAT':[2,3,4,5]},
        metadata_file_path: str = None,
        **kw: object,
    ):
        super().__init__(
            bags_per_slide=bags_per_slide,
            side_length=side_length,
            num_grids=num_grids,
            target=target,
            min_tiles=min_tiles,
            datasets_folds=datasets_folds,
            metadata_file_path=metadata_file_path,
            **kw,
        )
        self.features_dir = features_dir
        self._bag_size = (side_length**2)*num_grids

    def get_bag(self, item: int):
        slide = self._slides_manager.get_slide(item // self._instances_per_slide)
        slide_name = slide.slide_context.image_file_name
        h5_file_name = f"{slide_name}_features.h5"

        with h5py.File(Path(join(self.features_dir, h5_file_name)), "r") as h5_file:
            slide_features = h5_file["features"][:]
            slide_coords = h5_file["coords"][:]

        patch_extractor = self.patch_extractor_constructor(slide=slide)
        bag_coords = patch_extractor._pixels_to_extract.astype("int32")

        features_dim = slide_features.shape[1]
        bag = np.zeros((self._bag_size, features_dim), dtype="float32")

        slide_coords_str = np.unique(np.array([np.array2string(x) for x in slide_coords.astype("int32")]))
        bag_coords_str = np.unique(np.array([np.array2string(x) for x in bag_coords.astype("int32")]))
        # print(len(slide_coords_str))
        # print(len(bag_coords_str))
        # print(bag_coords_str)

        existing_features_indices_in_slide = np.where(
            np.isin(slide_coords_str, bag_coords_str)
        )[0]
        existing_features_indices_in_bag = np.where(
            np.isin(bag_coords_str, slide_coords_str)
        )[0]

        assert len(existing_features_indices_in_slide) == len(
            existing_features_indices_in_bag
        ), f"{len(existing_features_indices_in_slide)=} != {len(existing_features_indices_in_bag)=}"

        bag[existing_features_indices_in_bag] = slide_features[
            existing_features_indices_in_slide
        ]

        bag = torch.from_numpy(bag)

        label = slide.slide_context.get_biomarker_value(bio_marker=self._target)
        if label == "Positive":
            label = 1
        elif label == "Negative":
            label = 0
        label = torch.tensor(label)

        return {
            "features": bag,
            "coords": bag_coords,
            "label": label,
            "slide_name": slide_name,
        }


class SlideRandomFeaturesDataset(SlideRandomDataset):
    def __init__(
        self,
        features_dir: str,
        bags_per_slide: int = 1,
        bag_size=100,
        target: str = "er_status",
        min_tiles: int = 100,
        datasets_folds: Dict = {'CAT':[2,3,4,5]},
        metadata_file_path: str = None,
        **kw: object,
    ):
        super().__init__(
            bags_per_slide=bags_per_slide,
            bag_size=bag_size,
            target=target,
            min_tiles=min_tiles,
            datasets_folds=datasets_folds,
            metadata_file_path=metadata_file_path,
            **kw,
        )
        self.features_dir = features_dir

    def get_bag(self, item: int):
        slide = self._slides_manager.get_slide(item // self._instances_per_slide)
        slide_name = slide.slide_context.image_file_name
        h5_file_name = f"{slide_name}_features.h5"

        with h5py.File(Path(join(self.features_dir, h5_file_name)), "r") as h5_file:
            slide_features = h5_file["features"][:]
            slide_coords = h5_file["coords"][:]

        tile_indecies = np.random.choice(len(slide_features), self._bag_size)
        bag_coords = np.array([slide_coords[idx] for idx in tile_indecies])

        features_dim = slide_features.shape[1]
        bag = np.zeros((self._bag_size, features_dim), dtype="float32")

        bag = slide_features[tile_indecies]

        bag, bag_coords = torch.from_numpy(bag), torch.from_numpy(bag_coords)

        label = slide.slide_context.get_biomarker_value(bio_marker=self._target)
        label = torch.tensor(label)

        return {
            "features": bag,
            "coords": bag_coords,
            "label": label,
            "slide_name": slide_name,
        }


class SlideStridedFeaturesDataset(SlideStridedDataset):
    def __init__(
        self,
        features_dir: str,
        bag_size=100,
        target: str = "er_status",
        min_tiles: int = 100,
        datasets_folds: Dict = {'CAT':[2,3,4,5]},
        metadata_file_path: str = None,
        **kw: object,
    ):
        super().__init__(
            bag_size=bag_size,
            target=target,
            min_tiles=min_tiles,
            datasets_folds=datasets_folds,
            metadata_file_path=metadata_file_path,
            **kw,
        )
        print(features_dir)
        self.features_dir = features_dir

    def get_bag(self, item: int):
        slide = self._slides_manager.get_slide(item // self._instances_per_slide)
        slide_name = slide.slide_context.image_file_name
        h5_file_name = f"{slide_name}_features.h5"

        with h5py.File(Path(join(self.features_dir, h5_file_name)), "r") as h5_file:
            slide_features = h5_file["features"][:]
            slide_coords = h5_file["coords"][:]

        stride = self.patch_extractor_constructor(slide)._stride
        tile_indecies = [(stride * i) % slide.tiles_count for i in range(self._bag_size)]
        bag_coords = np.array([slide_coords[idx] for idx in tile_indecies])

        features_dim = slide_features.shape[1]
        bag = np.zeros((self._bag_size, features_dim), dtype="float32")

        bag = slide_features[tile_indecies]

        bag, bag_coords = torch.from_numpy(bag), torch.from_numpy(bag_coords)

        label = slide.slide_context.get_biomarker_value(bio_marker=self._target)
        label = torch.tensor(label)

        return {
            "features": bag,
            "coords": bag_coords,
            "label": label,
            "slide_name": slide_name,
        }