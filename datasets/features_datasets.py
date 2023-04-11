from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch

from datasets.datasets import SlideGridDataset


class SlideGridFeaturesDataset(SlideGridDataset):
    def __init__(
        self,
        features_dir: str,
        bags_per_slide: int = 1,
        side_length=8,
        target: str = "ER",
        min_tiles: int = 100,
        dataset: str = "CAT",
        metadata_file_path: str = None,
        train: bool = True,  # controls folds
        val_fold: Optional[int] = None,
        **kw: object,
    ):
        super().__init__(
            bags_per_slide=bags_per_slide,
            side_length=side_length,
            target=target,
            min_tiles=min_tiles,
            dataset=dataset,
            metadata_file_path=metadata_file_path,
            val_fold=val_fold,
            train=train,
            **kw,
        )
        self.features_dir = features_dir
        self.bag_size = side_length**2

    def get_bag(self, item: int):
        slide = self._slides_manager.get_slide(item % self.num_slides)
        slide_name = slide.slide_context.image_file_name_stem
        h5_file_name = f"{slide_name}_features.h5"

        with h5py.File(Path(self.features_dir) / h5_file_name, "r") as h5_file:
            slide_features = h5_file["features"][:]
            slide_coords = h5_file["coords"][:]

        patch_extractor = self.patch_extractor_constructor(slide=slide)
        bag_coords = patch_extractor._pixels_to_extract.astype("int32")

        features_dim = slide_features.shape[1]
        bag = np.zeros((self.bag_size, features_dim), dtype="float32")

        slide_coords_str = np.array([np.array2string(x) for x in slide_coords])
        bag_coords_str = np.array([np.array2string(x) for x in bag_coords])

        existing_features_indices_in_slide = np.where(
            np.isin(slide_coords_str, bag_coords_str)
        )[0]
        existing_features_indices_in_bag = np.where(
            np.isin(bag_coords_str, slide_coords_str)
        )[0]

        assert len(existing_features_indices_in_slide) == len(
            existing_features_indices_in_bag
        )

        bag[existing_features_indices_in_bag] = slide_features[
            existing_features_indices_in_slide
        ]

        bag, bag_coords = torch.from_numpy(bag), torch.from_numpy(bag_coords)

        label = slide.slide_context.get_biomarker_value(bio_marker=self._target)
        if label == "Positive":
            label = 1
        elif label == "Negative":
            label = 0
        label = torch.tensor(label).expand(self._bag_size).clone()

        return {
            "features": bag,
            "coords": bag_coords,
            "label": label,
            "slide_name": slide_name,
        }
