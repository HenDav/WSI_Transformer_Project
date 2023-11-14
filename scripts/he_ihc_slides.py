import os
from pathlib import Path
from typing import List, Dict
from enum import Enum
from dataclasses import dataclass
import pandas as pd
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter


class SlideType(Enum):
    HE = 1
    IHC = 2


class SlidesMapping:
    def __init__(self, paths: List[Path], slide_type: SlideType):
        self._paths = paths
        self._slide_type = slide_type
        self._block_id_to_thumb_ids: Dict[str, List[str]] = {}
        self._thumb_id_to_path: Dict[str, Path] = {}
        self._block_id_to_paths: Dict[str, List[Path]] = {}
        for path in paths:
            thumb_id = SlidesMapping._get_thumb_id(path=path)
            block_id = SlidesMapping._get_block_id(thumb_id=thumb_id)

            if block_id not in self._block_id_to_thumb_ids:
                self._block_id_to_thumb_ids[block_id] = []

            self._block_id_to_thumb_ids[block_id].append(thumb_id)
            self._thumb_id_to_path[thumb_id] = path

            if block_id not in self._block_id_to_paths:
                self._block_id_to_paths[block_id] = []
            self._block_id_to_paths[block_id].append(path)

    @property
    def slide_type(self) -> SlideType:
        return self._slide_type

    @property
    def block_id_to_thumb_ids(self) -> Dict[str, List[str]]:
        return self._block_id_to_thumb_ids

    @property
    def thumb_id_to_path(self) -> Dict[str, Path]:
        return self._thumb_id_to_path

    @property
    def block_id_to_paths(self) -> Dict[str, List[Path]]:
        return self._block_id_to_paths

    @staticmethod
    def _get_thumb_id(path: Path) -> str:
        parts = path.stem.split('thumb', 1)
        thumb_id = 'thumb' + parts[1]
        return thumb_id

    @staticmethod
    def _get_block_id(thumb_id: str) -> str:
        return thumb_id.rsplit('_', 1)[0]


def list_files(paths: List[Path], pattern: str) -> List[Path]:
    listed_file_paths = []
    for path in paths:
        for root, dirs, files in os.walk(path):
            for file in files:
                if pattern in file and file.endswith('.jpg'):
                    full_path = os.path.normpath(os.path.join(root, file))
                    listed_file_paths.append(Path(full_path))
    return listed_file_paths


def get_thumb_id(path: Path) -> str:
    parts = path.stem.split('thumb', 1)
    thumb_id = 'thumb' + parts[1]
    return thumb_id


def get_block_id(thumb_id: str) -> str:
    return thumb_id.rsplit('_', 1)[0]


he_base_path = [
    Path('/mnt/gipmed_new/Data/Breast/Carmel/1-8/Batch_1/thumbs'),
    Path('/mnt/gipmed_new/Data/Breast/Carmel/1-8/Batch_2/thumbs'),
    Path('/mnt/gipmed_new/Data/Breast/Carmel/1-8/Batch_3/thumbs'),
    Path('/mnt/gipmed_new/Data/Breast/Carmel/1-8/Batch_4/thumbs'),
    Path('/mnt/gipmed_new/Data/Breast/Carmel/1-8/Batch_5/thumbs'),
    Path('/mnt/gipmed_new/Data/Breast/Carmel/1-8/Batch_6/thumbs'),
    Path('/mnt/gipmed_new/Data/Breast/Carmel/1-8/Batch_7/thumbs'),
    Path('/mnt/gipmed_new/Data/Breast/Carmel/1-8/Batch_8/thumbs'),
    Path('/mnt/gipmed_new/Data/Breast/Carmel/9-11/Batch_9/thumbs'),
    Path('/mnt/gipmed_new/Data/Breast/Carmel/9-11/Batch_10/thumbs'),
    Path('/mnt/gipmed_new/Data/Breast/Carmel/9-11/Batch_11/thumbs')
]

ihc_base_path = [
    Path('/mnt/gipmed_new/Data/Breast/Carmel/Her2/Batch_1/thumbs'),
    Path('/mnt/gipmed_new/Data/Breast/Carmel/Her2/Batch_2/thumbs'),
    Path('/mnt/gipmed_new/Data/Breast/Carmel/Her2/Batch_3/thumbs'),
    Path('/mnt/gipmed_new/Data/Breast/Carmel/Her2/Batch_4/thumbs'),
    Path('/mnt/gipmed_new/Data/Breast/Carmel/Her2/Batch_5/thumbs'),
    Path('/mnt/gipmed_new/Data/Breast/Carmel/Her2/Batch_6/thumbs'),
]

# ihc_base_path = Path('C:/slide_thumbs/IHC')
he_paths = list_files(path=he_base_path, pattern='thumb')
ihc_paths = list_files(path=ihc_base_path, pattern='thumb')

ihc_slides_mappings = SlidesMapping(paths=ihc_paths, slide_type=SlideType.IHC)
he_slides_mappings = SlidesMapping(paths=he_paths, slide_type=SlideType.HE)

block_ids = list(set(list(he_slides_mappings.block_id_to_thumb_ids.keys()) + list(ihc_slides_mappings.block_id_to_thumb_ids.keys())))

list_of_paths = []
for block_id in block_ids:
    # he_thumb_ids = he_slides_mappings.block_id_to_thumb_ids[block_id]
    # ihc_thumb_ids = ihc_slides_mappings.block_id_to_thumb_ids[block_id]
    # current_he_paths = [str(he_slides_mappings.thumb_id_to_path[thumb_id]) for thumb_id in he_thumb_ids]
    try:
        current_he_paths = he_slides_mappings.block_id_to_paths[block_id]
        current_ihc_paths = ihc_slides_mappings.block_id_to_paths[block_id]
        current_paths = current_he_paths + current_ihc_paths
        list_of_paths.append(current_paths)
    except:
        pass

df = pd.DataFrame(list_of_paths)
filename = "output.xlsx"
df.to_excel(filename, index=False, engine='openpyxl')

# Load the Excel file
workbook = load_workbook(filename)
sheet = workbook.active

# Adjust column widths
for column in sheet.columns:
    max_length = 0
    column_letter = get_column_letter(column[0].column)
    for cell in column:
        try:
            if len(str(cell.value)) > max_length:
                max_length = len(cell.value)
        except:
            pass
    adjusted_width = (max_length + 2)
    sheet.column_dimensions[column_letter].width = adjusted_width

# Save the changes
workbook.save(filename)