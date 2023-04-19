import math
import os
import pickle
from abc import ABC, abstractmethod
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Dict, List, Optional, cast

import cv2
import h5py
import numpy as np

# openslide
# OPENSLIDE_PATH = r'C:\openslide-win64-20220811\bin'
# if hasattr(os, 'add_dll_directory'):
#     # Python >= 3.8 on Windows
#     with os.add_dll_directory(OPENSLIDE_PATH):
#         import openslide
# else:
import openslide
import pandas
import torch
from matplotlib import pyplot as plt
from PIL import Image

from wsi_core import constants, utils
from wsi_core.base import SeedableObject


class BioMarker(Enum):
    ER = auto()
    PR = auto()
    HER2 = auto()


BIOMARKER_TO_COLUMN = {
    BioMarker.ER: constants.er_status_column_name,
    BioMarker.PR: constants.pr_status_column_name,
    BioMarker.HER2: constants.her2_status_column_name,
}


class SlideContext:
    def __init__(
        self,
        row_index: int,
        metadata: pandas.DataFrame,
        dataset_paths: Dict[str, Path],
        desired_mpp: int,
        tile_size: int,
    ):
        self._row_index = row_index
        self._row = metadata.iloc[[row_index]]
        self._dataset_path = dataset_paths[
            self._row[constants.dataset_id_column_name].item()
        ]
        self._desired_mpp = desired_mpp
        self._tile_size = tile_size
        self._image_file_name = self._row[constants.file_column_name].item()
        self._image_file_path = self._dataset_path / self._image_file_name
        self._dataset_id = self._row[constants.dataset_id_column_name].item()
        self._image_file_name_stem = self._image_file_path.stem
        self._image_file_name_suffix = self._image_file_path.suffix
        self._orig_mpp = self._row[constants.mpp_column_name].item()
        if math.isnan(self._orig_mpp):
            self._orig_mpp = 0.25  # placeholder value, this should be dealt with when creating the metadata.
        self._curr_mpp = constants.current_mpp
        self._legitimate_tiles_count = self._row[
            constants.legitimate_tiles_column_name
        ].item()
        self._fold = self._row[constants.fold_column_name].item()
        self._downsample_from_curr = self._desired_mpp / self._curr_mpp
        self._downsample_from_orig = utils.round_to_nearest_power_of_two(
            self._desired_mpp / self._orig_mpp
        )
        self._zero_level_tile_size = self._tile_size * self._downsample_from_orig
        self._er = self._row[constants.er_status_column_name].item()
        self._pr = self._row[constants.pr_status_column_name].item()
        self._her2 = self._row[constants.her2_status_column_name].item()
        self._color_channels = 3
        self._is_h5 = "/h5/" in str(self._dataset_path)
        if self._is_h5:
            self.read_region_around_pixel = self._read_region_around_pixel_h5
        else:
            self.read_region_around_pixel = self._read_region_around_pixel_openslide
            self._slide = openslide.open_slide(self._image_file_path)
            self._level, self._level_downsample = self._get_best_level_for_downsample(
                slide=self._slide
            )
            self._selected_level_tile_size = self._tile_size * self._level_downsample

    @property
    def row_index(self) -> int:
        return self._row_index

    @property
    def dataset_path(self) -> Path:
        return self._dataset_path

    @property
    def desired_magnification(self) -> int:
        return round((self._desired_mpp**-1.0) * 10)

    @property
    def image_file_name(self) -> str:
        return self._image_file_name

    @property
    def image_file_path(self) -> Path:
        return self._image_file_path

    @property
    def image_file_name_stem(self) -> str:
        return self._image_file_name_stem

    @property
    def dataset_id(self) -> str:
        return self._dataset_id

    # @property
    # def slide(self) -> openslide.OpenSlide:
    #     return self._slide

    # @property
    # def level(self) -> int:
    #     return self._level

    @property
    def tile_size(self) -> int:
        return self._tile_size

    # @property
    # def selected_level_tile_size(self) -> int:
    #     return self._selected_level_tile_size
    #
    # @property
    # def selected_level_half_tile_size(self) -> int:
    #     return self._selected_level_tile_size // 2

    @property
    def zero_level_tile_size(self) -> int:
        return self._zero_level_tile_size

    @property
    def zero_level_half_tile_size(self) -> int:
        return self._zero_level_tile_size // 2

    @property
    def zero_level_tile_size_x_offset(self) -> int:
        return np.array([self._zero_level_tile_size, 0])

    @property
    def zero_level_tile_size_y_offset(self) -> int:
        return np.array([0, self._zero_level_tile_size])

    @property
    def zero_level_tile_size_x_offset(self) -> int:
        return np.array([self._zero_level_tile_size, 0])

    @property
    def zero_level_tile_size_y_offset(self) -> int:
        return np.array([0, self._zero_level_tile_size])

    @property
    def orig_mpp(self) -> float:
        return self._orig_mpp

    @property
    def curr_mpp(self) -> float:
        return self._curr_mpp

    @property
    def er(self) -> bool:
        return self._er

    @property
    def pr(self) -> bool:
        return self._pr

    @property
    def her2(self) -> bool:
        return self._her2

    def mm_to_pixels(self, mm: float) -> int:
        pixels = int(mm / (self._curr_mpp / 1000))
        return pixels

    def pixels_to_locations(self, pixels: np.ndarray) -> np.ndarray:
        return pixels // self._zero_level_tile_size

    def locations_to_pixels(self, locations: np.ndarray) -> np.ndarray:
        return (locations * self._zero_level_tile_size).astype(np.int64)

    # def read_region_around_pixel(self, pixel: np.ndarray) -> Image:
    #     return self._read_region_around_pixel_h5(pixel=pixel)

    def _read_region_around_pixel_openslide(self, pixel: np.ndarray) -> Image:
        # slide = openslide.open_slide(self._image_file_path)
        # level, level_downsample = self._get_best_level_for_downsample(slide=slide)
        # selected_level_tile_size = self._tile_size * level_downsample
        slide = self._slide
        level, _ = self._level, self._level_downsample
        selected_level_tile_size = self._selected_level_tile_size
        top_left_pixel = (pixel - self.zero_level_half_tile_size).astype(int)
        top_left_pixel[0], top_left_pixel[1] = top_left_pixel[1], top_left_pixel[0]
        region = slide.read_region(
            top_left_pixel, level, (selected_level_tile_size, selected_level_tile_size)
        ).convert("RGB")
        if selected_level_tile_size != self.tile_size:
            region = region.resize((self.tile_size, self.tile_size))
        return region

    def _np_to_h5_key(self, coords: np.ndarray) -> str:
        coords = coords.astype(int)
        key = str((coords[0], coords[1]))
        return key

    # TODO enable larger sample & sample at other mpp.
    def _read_region_around_pixel_h5(self, pixel: np.ndarray) -> Image:
        pixel = pixel // self._downsample_from_orig
        with h5py.File(f"{self._image_file_path}.h5", "r") as file:
            tile_size = self._tile_size
            x_offset = np.array([tile_size, 0])
            y_offset = np.array([0, tile_size])
            top_left_pixel = (pixel - 1 / 2 * (x_offset + y_offset)).astype(int)
            local_coords = top_left_pixel % tile_size
            top_left_coords = top_left_pixel - local_coords

            image = np.zeros((self._tile_size, self._tile_size, self._color_channels))
            filled_area = 0

            if self._np_to_h5_key(top_left_coords) in file["tiles"].keys():
                image[
                    : (self._tile_size - local_coords[0]),
                    : (self._tile_size - local_coords[1]),
                    :,
                ] = file["tiles"][self._np_to_h5_key(top_left_coords)]["array"][
                    local_coords[0] :, local_coords[1] :, :
                ]
                filled_area += (self._tile_size - local_coords[0]) * (
                    self._tile_size - local_coords[1]
                )
            # else:
            #     print(
            #         f"took blank part of patch since {top_left_coords} didn't correspond to valid patch"
            #     )

            if filled_area == self._tile_size**2:
                return Image.fromarray(np.uint8(image), mode="RGB")

            if self._np_to_h5_key(top_left_coords + x_offset) in file["tiles"].keys():
                image[
                    (self._tile_size - local_coords[0]) :,
                    : (self._tile_size - local_coords[1]),
                    :,
                ] = file["tiles"][self._np_to_h5_key(top_left_coords + x_offset)][
                    "array"
                ][
                    : local_coords[0], local_coords[1] :, :
                ]
                filled_area += local_coords[0] * (self._tile_size - local_coords[1])
            # else:
            #     print(
            #         f"took blank part of patch since {top_left_coords + x_offset} didn't correspond to valid patch"
            #     )

            if filled_area == self._tile_size**2:
                return Image.fromarray(np.uint8(image), mode="RGB")

            if self._np_to_h5_key(top_left_coords + y_offset) in file["tiles"].keys():
                image[
                    : (self._tile_size - local_coords[0]),
                    (self._tile_size - local_coords[1]) :,
                    :,
                ] = file["tiles"][self._np_to_h5_key(top_left_coords + y_offset)][
                    "array"
                ][
                    local_coords[0] :, : local_coords[1], :
                ]
                filled_area += (self._tile_size - local_coords[0]) * local_coords[1]
            # else:
            #     print(
            #         f"took blank part of patch since {top_left_coords + y_offset} didn't correspond to valid patch"
            #     )

            if filled_area == self._tile_size**2:
                return Image.fromarray(np.uint8(image), mode="RGB")

            if (
                self._np_to_h5_key(top_left_coords + x_offset + y_offset)
                in file["tiles"].keys()
            ):
                image[
                    (self._tile_size - local_coords[0]) :,
                    (self._tile_size - local_coords[1]) :,
                    :,
                ] = file["tiles"][
                    self._np_to_h5_key(top_left_coords + x_offset + y_offset)
                ][
                    "array"
                ][
                    : local_coords[0], : local_coords[1], :
                ]
                filled_area += local_coords[0] * local_coords[1]
            # else:
            #     print(
            #         f"took blank part of patch since {top_left_coords + x_offset + y_offset} didn't correspond to valid patch"
            #     )

        return Image.fromarray(np.uint8(image), mode="RGB")

    def get_biomarker_value(self, bio_marker: BioMarker) -> bool:
        if bio_marker is BioMarker.ER:
            return self._er
        elif bio_marker is BioMarker.PR:
            return self._pr
        elif bio_marker is BioMarker.HER2:
            return self._her2

    def _get_best_level_for_downsample(self, slide: openslide.OpenSlide):
        level = 0
        level_downsample = self._downsample_from_orig
        if self._downsample_from_orig > 1:
            for i, downsample in enumerate(slide.level_downsamples):
                if math.isclose(self._downsample_from_orig, downsample, rel_tol=1e-3):
                    level = i
                    level_downsample = 1
                    break
                elif downsample < self._downsample_from_orig:
                    level = i
                    level_downsample = int(
                        self._downsample_from_orig / slide.level_downsamples[level]
                    )

        # A tile of size (tile_size, tile_size) in an image downsampled by 'level_downsample', will cover the same image portion of a tile of size (adjusted_tile_size, adjusted_tile_size) in the original image
        return level, level_downsample


class SlideElement(SeedableObject):
    # __slots__ = ['_slide_context']

    def __init__(self, slide_context: SlideContext, **kw: object):
        super(SlideElement, self).__init__(**kw)
        self._slide_context = slide_context

    @property
    def slide_context(self) -> SlideContext:
        return self._slide_context


class Patch(SlideElement):
    # __slots__ = ['_center_pixel', '_image']

    def __init__(self, slide_context: SlideContext, center_pixel: np.ndarray):
        super().__init__(slide_context=slide_context)
        center_pixel = center_pixel.flatten()
        self._center_pixel = center_pixel
        self._image = None

    @property
    def image(self) -> torch.Tensor:
        # if self._image is None:
        self._image = self._slide_context.read_region_around_pixel(
            pixel=self._center_pixel
        )

        # image_tensor = torch.Tensor(self._image).permute([2, 0, 1])
        # return image_tensor
        return self._image

    @property
    def center_pixel(self) -> np.ndarray:
        return self._center_pixel

    # def get_white_ratio(self, white_intensity_threshold: int) -> float:
    #     patch_grayscale = self._image.convert('L')
    #     hist, _ = np.histogram(a=patch_grayscale, bins=self._slide_context.tile_size)
    #     white_ratio = np.sum(hist[white_intensity_threshold:]) / (self._slide_context.tile_size * self._slide_context.tile_size)
    #     return white_ratio


class Tile(Patch):
    # __slots__ = ['_location', '_top_left_pixel', '_center_pixel']

    def __init__(self, slide_context: SlideContext, top_left_pixel: np.ndarray):
        top_left_pixel = top_left_pixel.flatten()
        self._top_left_pixel = top_left_pixel.astype(np.int64)
        self._center_pixel = top_left_pixel + slide_context.zero_level_half_tile_size
        super().__init__(slide_context=slide_context, center_pixel=self._center_pixel)

    def __hash__(self):
        return hash(self._location.tobytes())

    def __eq__(self, other):
        return self._location.tobytes() == other.tile_location.tobytes()

    def __ne__(self, other):
        return not (self == other)

    @property
    def tile_location(self) -> np.ndarray:
        return self._location

    @property
    def center_pixel(self) -> np.ndarray:
        return self._center_pixel

    @property
    def top_left_pixel(self) -> np.ndarray:
        return self._top_left_pixel

    def get_random_pixel(self) -> np.ndarray:
        offset = (
            self._slide_context.zero_level_half_tile_size * np.random.uniform(size=2)
        ).astype(np.int64)
        pixel = self._center_pixel + offset
        return pixel


class TilesManager(ABC, SlideElement):
    # __slots__ = ['_pixels', '_locations', '_tiles', '_location_to_tile', '_interior_tiles']

    tile_index = "tile_index"
    pixel_x = "pixel_x"
    pixel_y = "pixel_y"
    location_x = "location_x"
    location_y = "location_y"
    is_interior = "is_interior"

    def __init__(self, slide_context: SlideContext, pixels: Optional[np.array]):
        super().__init__(slide_context=slide_context)

        self._load_pixels = (
            self._load_pixels_h5
            if slide_context._is_h5
            else self._load_pixels_openslide
        )
        self._tiles_df = self._create_tiles_dataframe(pixels=pixels)

        # if pixels is None:
        #     self._pixels = self._load_pixels()
        # else:
        #     self._pixels = pixels
        #
        # self._locations = slide_context.pixels_to_locations(pixels=self._pixels)
        #
        # self._locations_df = pandas.DataFrame({"locations": [self._locations]})
        #
        # # self._tiles = self._create_tiles_list()
        # self._location_to_tile = self._create_tiles_dict()
        # # self._interior_tiles = self._create_interior_tiles_list()

    @property
    def tiles_count(self) -> int:
        return self._tiles_df.shape[0]

    @property
    def interior_tiles_count(self) -> int:
        return self._get_interior_tile_rows().shape[0]

    @property
    def tiles(self) -> List[Tile]:
        locations = TilesManager._get_locations_from_dataframe(df=self._tiles_df)
        return self._tiles_from_locations(locations=locations)

    @property
    def interior_tiles(self) -> List[Tile]:
        interior_tile_rows = self._get_interior_tile_rows()
        locations = TilesManager._get_locations_from_dataframe(df=interior_tile_rows)
        return self._tiles_from_locations(locations=locations)

    @property
    def has_interior_tiles(self) -> bool:
        return self.interior_tiles_count > 0

    def get_tiles_ratio(self, ratio: float) -> List[Tile]:
        modulo = int(1 / ratio)
        return self.get_tiles_modulo(modulo=modulo)

    def get_tiles_modulo(self, modulo: int) -> List[Tile]:
        return self.tiles[::modulo]

    def get_tile(self, tile_index: int) -> Tile:
        row = self._tiles_df.iloc[[tile_index]]
        top_left_pixel = row[[TilesManager.pixel_x, TilesManager.pixel_x]].to_numpy()
        return Tile(slide_context=self._slide_context, top_left_pixel=top_left_pixel)

    def get_interior_tile(self, tile_index: int) -> Tile:
        interior_tile_rows = self._get_interior_tile_rows()
        row = interior_tile_rows.iloc[[tile_index]]
        top_left_pixel = row[[TilesManager.pixel_x, TilesManager.pixel_x]].to_numpy()
        return Tile(slide_context=self._slide_context, top_left_pixel=top_left_pixel)

    def get_random_tile(self) -> Tile:
        tile_index = self._rng.integers(low=0, high=self.tiles_count)
        return self.get_tile(tile_index=tile_index)

    def get_random_interior_tile(self) -> Tile:
        tile_index = self._rng.integers(low=0, high=self.interior_tiles_count)
        return self.get_interior_tile(tile_index=tile_index)

    def get_random_pixel(self) -> np.ndarray:
        tile = self.get_random_tile()
        return tile.get_random_pixel()

    def get_random_interior_pixel(self) -> np.ndarray:
        tile = self.get_random_interior_tile()
        return tile.get_random_pixel()

    def get_tile_at_pixel(self, pixel: np.ndarray) -> Optional[Tile]:
        location = self._slide_context.pixels_to_locations(pixels=pixel)
        row = self._tiles_df.loc[
            (self._tiles_df[TilesManager.pixel_x] == location[0])
            & (self._tiles_df[TilesManager.pixel_y] == location[1])
        ]
        return self._location_to_tile.get(location.tobytes())

    def is_interior_tile(self, tile: Tile) -> bool:
        for i in range(3):
            for j in range(3):
                current_tile_location = tile.tile_location + np.array([i, j])
                if not self._is_valid_tile_location(
                    tile_location=current_tile_location
                ):
                    return False
        return True

    def _get_interior_tile_rows(self) -> pandas.DataFrame:
        return self._tiles_df.loc[self._tiles_df[TilesManager.is_interior] == 1]

    @staticmethod
    def _locations_to_bytes_list(locations: np.ndarray) -> List[bytes]:
        bytes_list = []
        for location in locations:
            bytes_list.append(location.tobytes())

        return bytes_list

    @staticmethod
    def _is_interior_location(
        location: np.ndarray, locations_bytes_list: List[bytes]
    ) -> bool:
        # for i in range(3):
        #     for j in range(3):
        #         adjacent_location = location + np.array([i, j])
        #         if adjacent_location.tobytes() not in locations_bytes_list:
        #             return False

        return True

    @staticmethod
    def _flag_interior_locations(locations: np.ndarray) -> np.ndarray:
        interior_location_flags = np.zeros(shape=locations.shape[0])
        locations_bytes_list = TilesManager._locations_to_bytes_list(
            locations=locations
        )
        for index, location in enumerate(locations):
            interior_location_flags[index] = TilesManager._is_interior_location(
                location=location, locations_bytes_list=locations_bytes_list
            )

        return interior_location_flags

    @staticmethod
    def _get_locations_from_dataframe(df: pandas.DataFrame) -> np.ndarray:
        return df[[TilesManager.location_x, TilesManager.location_y]].to_numpy()

    def _tiles_from_locations(self, locations: np.ndarray) -> List[Tile]:
        return [
            Tile(slide_context=self._slide_context, location=locations[i])
            for i in range(locations.shape[0])
        ]

    def _create_tiles_dataframe(self, pixels: Optional[np.ndarray]) -> pandas.DataFrame:
        if pixels is None:
            pixels = self._load_pixels()

        locations = self._slide_context.pixels_to_locations(pixels=pixels)
        interior_locations_flags = TilesManager._flag_interior_locations(
            locations=locations
        )
        interior_locations_flags = np.expand_dims(interior_locations_flags, axis=1)
        tile_indices = np.array(list(range(locations.shape[0])))
        tile_indices = np.expand_dims(tile_indices, axis=1)
        data = np.concatenate(
            (tile_indices, pixels, locations, interior_locations_flags), axis=1
        ).astype(np.int32)
        tiles_df = pandas.DataFrame(
            data=data,
            columns=[
                TilesManager.tile_index,
                TilesManager.pixel_x,
                TilesManager.pixel_y,
                TilesManager.location_x,
                TilesManager.location_y,
                TilesManager.is_interior,
            ],
        )
        return tiles_df

    def _load_pixels_h5(self) -> np.ndarray:
        with h5py.File(
            f"{self._slide_context._image_file_path}.h5", "r"
        ) as file_handle:
            pixels = file_handle["segmentation_pixels"][...]
            pixels[:, 0], pixels[:, 1] = pixels[:, 1], pixels[:, 0].copy()

        return pixels

    def _load_pixels_openslide(self) -> np.ndarray:
        return utils.load_segmentation_data(
            dataset_path=self._slide_context.dataset_path,
            desired_magnification=self._slide_context.desired_magnification,
            image_file_name_stem=self._slide_context.image_file_name_stem,
            tile_size=self._slide_context.tile_size)
        # grid_file_path = os.path.normpath(
        #     os.path.join(
        #         self._slide_context.dataset_path,
        #         f"Grids_{self._slide_context.desired_magnification}",
        #         f"{self._slide_context.image_file_name_stem}--tlsz{self._slide_context.tile_size}.data",
        #     )
        # )
        # with open(grid_file_path, "rb") as file_handle:
        #     pixels = np.array(pickle.load(file_handle))
        #
        # return pixels

    # def _create_tiles_list(self) -> List[Tile]:
    #     tiles = []
    #     for i in range(self._locations.shape[0]):
    #         tiles.append(Tile(slide_context=self._slide_context, location=self._locations[i, :]))
    #     return tiles
    #
    # def _create_interior_tiles_list(self) -> List[Tile]:
    #     interior_tiles = []
    #     for tile in self._tiles:
    #         if self.is_interior_tile(tile=tile):
    #             interior_tiles.append(tile)
    #
    #     return interior_tiles
    #
    # def _create_tiles_dict(self) -> Dict[bytes, int]:
    #     tiles_dict = {}
    #     for index, location in enumerate(self._locations):
    #         tiles_dict[location.tobytes()] = index
    #
    #     return tiles_dict

    def _is_valid_tile_location(self, tile_location: np.ndarray) -> bool:
        if tile_location.tobytes() in self._location_to_tile:
            return True
        return False

    # def _pixels_to_locations(self, pixels: np.ndarray) -> np.ndarray:
    #     return (pixels / self._slide_context.zero_level_tile_size).astype(np.int64)
    #
    # def _locations_to_pixels(self, locations: np.ndarray) -> np.ndarray:
    #     return (locations * self._slide_context.zero_level_tile_size).astype(np.int64)


class ConnectedComponent(TilesManager):
    # __slots__ = ['_top_left_tile_location', '_bottom_right_tile_location']

    def __init__(self, slide_context: SlideContext, pixels: np.ndarray):
        super().__init__(slide_context=slide_context, pixels=pixels)
        self._top_left_tile_location = np.array(
            [np.min(self._locations[:, 0]), np.min(self._locations[:, 1])]
        )
        self._bottom_right_tile_location = np.array(
            [np.max(self._locations[:, 0]), np.max(self._locations[:, 1])]
        )

    @property
    def top_left_tile_location(self) -> np.ndarray:
        return self._top_left_tile_location

    @property
    def bottom_right_tile_location(self) -> np.ndarray:
        return self._bottom_right_tile_location

    def calculate_bounding_box_aspect_ratio(self):
        box = (self.bottom_right_tile_location - self.top_left_tile_location) + 1
        return box[0] / box[1]


class Slide(TilesManager):
    # __slots__ = ['_min_component_ratio', '_max_aspect_ratio_diff', '_bitmap', '_components']

    def __init__(
        self,
        slide_context: SlideContext,
        min_component_ratio: float = 0.92,
        max_aspect_ratio_diff: float = 0.02,
    ):
        super().__init__(slide_context=slide_context, pixels=None)
        self._min_component_ratio = min_component_ratio
        self._max_aspect_ratio_diff = max_aspect_ratio_diff
        # self._bitmap = self._create_bitmap(plot_bitmap=False)
        # self._components = self._create_connected_components()

    @property
    def components(self) -> List[ConnectedComponent]:
        return self._components

    def get_component(self, component_index: int) -> ConnectedComponent:
        return self._components[component_index]

    def get_random_component(self) -> ConnectedComponent:
        component_index = int(np.random.randint(len(self._components), size=1))
        return self.get_component(component_index=component_index)

    def get_component_at_pixel(self, pixel: np.ndarray) -> Optional[ConnectedComponent]:
        tile_location = (pixel / self._slide_context.zero_level_tile_size).astype(
            np.int64
        )
        return self._location_to_tile.get(tile_location.tobytes())

    def _create_bitmap(self, plot_bitmap: bool = False) -> Image:
        # indices = (np.array(self._locations) / self._slide_context.zero_level_tile_size).astype(int)
        dim1_size = self._locations[:, 0].max() + 1
        dim2_size = self._locations[:, 1].max() + 1
        bitmap = np.zeros([dim1_size, dim2_size]).astype(int)

        for x, y in self._locations:
            bitmap[x, y] = 1

        bitmap = np.uint8(Image.fromarray((bitmap * 255).astype(np.uint8)))

        if plot_bitmap is True:
            plt.imshow(bitmap, cmap="gray")
            plt.show()

        return bitmap

    def _create_connected_component_from_bitmap(
        self, bitmap: np.ndarray
    ) -> ConnectedComponent:
        valid_tile_indices = np.where(bitmap)
        locations = np.array([valid_tile_indices[0], valid_tile_indices[1]]).transpose()
        pixels = self._slide_context.locations_to_pixels(locations=locations)
        return ConnectedComponent(slide_context=self._slide_context, pixels=pixels)

    def _create_connected_components(self) -> List[ConnectedComponent]:
        (
            components_count,
            components_labels,
            components_stats,
            _,
        ) = cv2.connectedComponentsWithStats(self._bitmap)
        components = []

        for component_id in range(1, components_count):
            bitmap = (components_labels == component_id).astype(int)
            connected_component = self._create_connected_component_from_bitmap(
                bitmap=bitmap
            )
            components.append(connected_component)

        components_sorted = sorted(
            components, key=lambda item: item.tiles_count, reverse=True
        )
        largest_component = components_sorted[0]
        largest_component_aspect_ratio = (
            largest_component.calculate_bounding_box_aspect_ratio()
        )
        largest_component_size = largest_component.tiles_count
        valid_components = [largest_component]
        for component in components_sorted[1:]:
            current_aspect_ratio = component.calculate_bounding_box_aspect_ratio()
            current_component_size = component.tiles_count
            if (
                np.abs(largest_component_aspect_ratio - current_aspect_ratio)
                < self._max_aspect_ratio_diff
            ) and (
                (current_component_size / largest_component_size)
                > self._min_component_ratio
            ):
                valid_components.append(component)

        return valid_components


class PatchExtractor(ABC):
    def __init__(
        self, slide: Slide, max_attempts: int = 10, slide_context: SlideContext = None
    ):
        self._slide = slide
        self._max_attempts = max_attempts
        self.slide_context = self._slide if not slide_context else slide_context

    @abstractmethod
    def _extract_center_pixel(self) -> np.ndarray:
        pass

    def extract_patch(
        self, patch_validators: List[Callable[[Patch], bool]]
    ) -> Optional[Patch]:
        attempts = 0
        while True:
            center_pixel = self._extract_center_pixel()
            if center_pixel is None:
                attempts = attempts + 1
                continue

            patch = Patch(slide_context=self.slide_context, center_pixel=center_pixel)

            patch_validation_failed = False
            for patch_validator in patch_validators:
                if not patch_validator(patch):
                    patch_validation_failed = True
                    break

            if patch_validation_failed is True:
                attempts = attempts + 1
                continue

            return patch, center_pixel
        return None


class RandomPatchExtractor(PatchExtractor):
    def __init__(self, slide: Slide):
        super().__init__(slide=slide)

    def _extract_center_pixel(self) -> np.ndarray:
        tile = self._slide.get_random_interior_tile()
        pixel = tile.get_random_pixel()
        return pixel


class StridedPatchExtractor(PatchExtractor):
    def __init__(self, slide: Slide, num_patches):
        super().__init__(slide=slide)
        self._num_patches = num_patches
        self._patches_so_far = 0
        self._stride = max(self._slide.interior_tiles_count // self._num_patches, 1)

    def _extract_center_pixel(self) -> np.ndarray:
        if self._patches_so_far >= self._num_patches:
            raise Exception
        self._patches_so_far = self._patches_so_far + 1
        tile = self._slide.get_interior_tile(
            (self._stride * self._patches_so_far) % self._slide.interior_tiles_count
        )
        pixel = tile.center_pixel
        return pixel


class SinglePatchExtractor(PatchExtractor):
    def __init__(self, tile: Tile):
        super().__init__(slide=None, slide_context=tile.slide_context)
        self._tile = tile

    def _extract_center_pixel(self) -> np.ndarray:
        return self._tile.center_pixel


class GridPatchExtractor(PatchExtractor):
    def __init__(self, slide: Slide, side_length):
        super().__init__(slide=slide)
        self._num_patches = side_length**2
        self._side_length = side_length
        self._patches_so_far = 0
        self._pixels_to_extract = np.zeros((self._num_patches, 2))
        tile = self._slide.get_random_interior_tile()
        center_pixel = tile.center_pixel
        top_left_tile_center_pixel = center_pixel - (
            tile._slide_context.zero_level_tile_size
            * (self._side_length - int(side_length % 2 != 0))
            // 2
        )

        for i in range(self._side_length):
            for j in range(self._side_length):
                self._pixels_to_extract[j * self._side_length + i] = (
                    top_left_tile_center_pixel
                    + tile._slide_context.zero_level_tile_size_x_offset * j
                    + tile._slide_context.zero_level_tile_size_y_offset * i
                )

    def _extract_center_pixel(self) -> np.ndarray:
        if self._patches_so_far >= self._num_patches:
            raise Exception
        self._patches_so_far += 1
        return self._pixels_to_extract[self._patches_so_far - 1]


class ProximatePatchExtractor(PatchExtractor):
    _max_attempts = 10

    def __init__(self, slide: Slide, reference_patch: Patch, inner_radius_mm: float):
        super().__init__(slide=slide)
        self._reference_patch = reference_patch
        self._inner_radius_mm = inner_radius_mm
        self._inner_radius_pixels = self._slide.slide_context.mm_to_pixels(
            mm=inner_radius_mm
        )

    def _extract_center_pixel(self) -> np.ndarray:
        pixel = self._reference_patch.center_pixel
        angle = 2 * np.pi * np.random.uniform(size=1)[0]
        direction = np.array([np.cos(angle), np.sin(angle)])
        proximate_pixel = (pixel + self._inner_radius_pixels * direction).astype(int)
        tile = self._slide.get_tile_at_pixel(pixel=proximate_pixel)
        if (tile is not None) and self._slide.is_interior_tile(tile=cast(Tile, tile)):
            return proximate_pixel

        return None
