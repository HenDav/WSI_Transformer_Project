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


class SlideData:
    def __init__(self, path: Path):
        self._path = path

    @property
    def slide_name(self) -> str:
        return self._path.stem

    @property
    def slide_id(self) -> str:
        return self.slide_name.replace('_', '/')

    @property
    def block_id(self) -> str:
        return self.slide_id.rsplit('/', 1)[0]

    @property
    def tissue_id(self) -> str:
        return self.block_id.rsplit('/', 1)[0]

    @property
    def sample_id(self) -> str:
        return self.tissue_id.rsplit('/', 1)[0]

    @property
    def sample_id(self) -> str:
        return self.tissue_id.rsplit('/', 1)[0]

    @property
    def batch_id(self) -> str:
        return self._path.parts[-2].capitalize()

    @property
    def dataset_name(self) -> str:
        batch_id = self.batch_id
        if batch_id.startswith('Carmel'):
            return 'Carmel'
        if batch_id.startswith('Her2'):
            return 'Her2'

    def get_row(self) -> List[str]:
        return [self.dataset_name, self.batch_id, self.slide_name, self.block_id, self.tissue_id, self.sample_id]

    @staticmethod
    def get_column_names() -> List[str]:
        return [
            'DatasetName',
            'BatchID',
            'SlideName',
            'SlideID',
            'BlockID',
            'TissueID',
        ]


class SlidesMapping:
    def __init__(self, paths: List[Path]):
        self._paths = paths
        self._slides_data: List[SlideData] = []
        self._block_id_to_slide_ids: Dict[str, List[str]] = {}
        self._slide_id_to_path: Dict[str, Path] = {}
        self._block_id_to_paths: Dict[str, List[Path]] = {}

        for path in paths:
            print(path)
            slide_data = SlideData(path=path)
            slide_id = slide_data.slide_id
            block_id = slide_data.block_id

            if block_id not in self._block_id_to_slide_ids:
                self._block_id_to_slide_ids[block_id] = []

            self._block_id_to_slide_ids[block_id].append(slide_id)
            self._slide_id_to_path[slide_id] = path

            if block_id not in self._block_id_to_paths:
                self._block_id_to_paths[block_id] = []
            self._block_id_to_paths[block_id].append(path)
            self._slides_data.append(slide_data)

    @property
    def block_id_to_slide_ids(self) -> Dict[str, List[str]]:
        return self._block_id_to_slide_ids

    @property
    def thumb_id_to_path(self) -> Dict[str, Path]:
        return self._slide_id_to_path

    @property
    def block_id_to_paths(self) -> Dict[str, List[Path]]:
        return self._block_id_to_paths

    @staticmethod
    def _get_block_id(thumb_id: str) -> str:
        return thumb_id.rsplit('_', 1)[0]

    def get_dataframe(self) -> pd.DataFrame:
        data = [slide_data.get_row() for slide_data in self._slides_data]
        columns = SlideData.get_column_names()
        df = pd.DataFrame(data, columns=columns)
        return df

    def save_dataframe(self, path: Path):
        path_str = str(path)
        df = self.get_dataframe()
        df.to_excel(path_str, index=False, engine='openpyxl')
        workbook = load_workbook(path_str)
        sheet = workbook.active

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

        workbook.save(path_str)


def list_files(paths: List[Path], ext: str) -> List[Path]:
    listed_file_paths = []
    for path in paths:
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(ext):
                    full_path = os.path.normpath(os.path.join(root, file))
                    listed_file_paths.append(Path(full_path))
    return listed_file_paths


def get_thumb_id(path: Path) -> str:
    parts = path.stem.split('thumb', 1)
    thumb_id = 'thumb' + parts[1]
    return thumb_id


def get_block_id(thumb_id: str) -> str:
    return thumb_id.rsplit('_', 1)[0]


base_paths = [
    Path('/mnt/gipmed_new/Data/Breast/Carmel/1-8/Batch_1/CARMEL1'),
    Path('/mnt/gipmed_new/Data/Breast/Carmel/1-8/Batch_2/CARMEL2'),
    Path('/mnt/gipmed_new/Data/Breast/Carmel/1-8/Batch_3/CARMEL3'),
    Path('/mnt/gipmed_new/Data/Breast/Carmel/1-8/Batch_4/CARMEL4'),
    Path('/mnt/gipmed_new/Data/Breast/Carmel/1-8/Batch_5/CARMEL5'),
    Path('/mnt/gipmed_new/Data/Breast/Carmel/1-8/Batch_6/CARMEL6'),
    Path('/mnt/gipmed_new/Data/Breast/Carmel/1-8/Batch_7/CARMEL7'),
    Path('/mnt/gipmed_new/Data/Breast/Carmel/1-8/Batch_8/CARMEL8'),
    Path('/mnt/gipmed_new/Data/Breast/Carmel/9-11/Batch_9/CARMEL9'),
    Path('/mnt/gipmed_new/Data/Breast/Carmel/9-11/Batch_10/CARMEL10'),
    Path('/mnt/gipmed_new/Data/Breast/Carmel/9-11/Batch_11/CARMEL11'),
    Path('/mnt/gipmed_new/Data/Breast/Carmel/Her2/Batch_1/Her2_1'),
    Path('/mnt/gipmed_new/Data/Breast/Carmel/Her2/Batch_2/HER2_2'),
    Path('/mnt/gipmed_new/Data/Breast/Carmel/Her2/Batch_3/HER2_3'),
    Path('/mnt/gipmed_new/Data/Breast/Carmel/Her2/Batch_4/HER2_4'),
    Path('/mnt/gipmed_new/Data/Breast/Carmel/Her2/Batch_5/HER2_5'),
    Path('/mnt/gipmed_new/Data/Breast/Carmel/Her2/Batch_6/HER2_6'),
]

# base_paths = [
#     Path('C:/slide_thumbs/IHC'),
#     Path('C:/slide_thumbs/HE')
# ]

slide_paths = list_files(paths=base_paths, ext='mrxs')
slides_mappings = SlidesMapping(paths=slide_paths)
slides_mappings.save_dataframe(path=Path('./output.xlsx'))


# block_ids = list(set(list(slides_mappings.block_id_to_slide_ids.keys())))
# list_of_paths = []
# for block_id in block_ids:
#     # he_thumb_ids = he_slides_mappings.block_id_to_thumb_ids[block_id]
#     # ihc_thumb_ids = ihc_slides_mappings.block_id_to_thumb_ids[block_id]
#     # current_he_paths = [str(he_slides_mappings.thumb_id_to_path[thumb_id]) for thumb_id in he_thumb_ids]
#     try:
#         current_he_paths = he_slides_mappings.block_id_to_paths[block_id]
#         current_ihc_paths = ihc_slides_mappings.block_id_to_paths[block_id]
#
#         current_he_slides = [f'{he_path.stem} (HE)' for he_path in current_he_paths]
#         current_ihc_slides = [f'{ihc_path.stem} (IHC)' for ihc_path in current_ihc_paths]
#
#         current_paths = current_he_slides + current_ihc_slides
#         list_of_paths.append(current_paths)
#     except:
#         pass
