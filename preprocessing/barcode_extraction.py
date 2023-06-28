# standard library
import os
from pathlib import Path
from typing import List
from tqdm import tqdm
from enum import Enum
from typing import Optional
import re
from dataclasses import dataclass

# pillow
from PIL import Image

# wsi
import core.constants
from core.constants import DatasetName

# openslide
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(core.constants.openslide_path):
        import openslide
else:
    import openslide

# pylibdmtx
from pylibdmtx.pylibdmtx import decode as decode


class BarcodeConvention(Enum):
    CARMEL = 1
    HAEMEK = 2
    NONE = 3


@dataclass
class SlideBarcode:
    slide_file_path: Path
    barcode_text: str
    barcode_image: Image


class BarcodeExtractor:
    def __init__(self, dataset_name: str, data_dir_path: Path, output_dir_path: Path):
        self._dataset_name = DatasetName[dataset_name]
        self._barcode_convention = self._get_barcode_convention()
        self._data_dir_path = data_dir_path
        self._output_dir_path = output_dir_path
        self._extensions = ['.mrxs', '.svs', '.tif', '.isyntax', '.ndpi']

    def _get_slide_paths(self) -> List[Path]:
        slide_paths = []
        for path in self._data_dir_path.glob(r'**/*'):
            if path.suffix in self._extensions:
                slide_paths.append(path)
        return slide_paths

    def _get_barcode_convention(self) -> BarcodeConvention:
        if self._dataset_name == DatasetName.BREAST_CARMEL or self._dataset_name == DatasetName.BREAST_CARMEL_BENIGN:
            return BarcodeConvention.CARMEL
        elif self._dataset_name == DatasetName.BREAST_HAEMEK:
            return BarcodeConvention.HAEMEK
        else:
            return BarcodeConvention.NONE

    @staticmethod
    def _check_barcode_validity(slide_barcode: str) -> bool:
        if re.search("\d{2}-\d+/\d+/\d+/\D{1}\Z", slide_barcode) is not None:
            return True
        else:
            return False

    @staticmethod
    def _adjust_barcode_using_convention(slide_barcode: str, barcode_convention: BarcodeConvention) -> str:
        if len(slide_barcode) < 10 or len(slide_barcode) > 16:
            return ''

        if barcode_convention == BarcodeConvention.CARMEL:
            if int(slide_barcode[:2]) > 90:  # old format, adjustment required
                year = int(slide_barcode[0:4]) - 9788
                if year > 200:  # year 21 shows as 201, and so on
                    year -= 180
                slide_barcode = str(year) + '-' + slide_barcode[4:]
        elif barcode_convention == BarcodeConvention.HAEMEK:
            year = int(slide_barcode[4]) + 14
            slide_barcode = str(year) + '-' + slide_barcode[5:]

        return slide_barcode

    @staticmethod
    def _get_slide_label_image(slide_path: Path) -> Optional[Image]:
        slide = openslide.OpenSlide(filename=str(slide_path))
        if 'label' in slide.associated_images._keys():
            return slide.associated_images['label']

        return None

    @staticmethod
    def _resample_slide_label_image(label_image: Image, downsmple_factor: float) -> Image:
        if downsmple_factor != 1:
            w, h = label_image.size
            label_image = label_image.resize(size=(int(w * downsmple_factor), int(h * downsmple_factor)), resample=Image.BICUBIC)
        return label_image

    @staticmethod
    def _try_extract_slide_barcode(slide_path: Path, downsmple_factor: float, barcode_convention: BarcodeConvention) -> SlideBarcode:
        validated_slide_barcodes = []
        label_image = BarcodeExtractor._get_slide_label_image(slide_path=slide_path)
        label_image = BarcodeExtractor._resample_slide_label_image(label_image=label_image, downsmple_factor=downsmple_factor)
        slide_barcodes = decode(image=label_image)
        for slide_barcode in slide_barcodes:
            slide_barcode = BarcodeExtractor._adjust_barcode_using_convention(slide_barcode=slide_barcode.data.decode('UTF-8'), barcode_convention=barcode_convention)
            if slide_barcode is not None and BarcodeExtractor._check_barcode_validity(slide_barcode):
                validated_slide_barcodes.append(slide_barcode)

        return validated_slide_barcodes

    @staticmethod
    def _extract_slide_barcode(slide_path: Path, barcode_convention: BarcodeConvention) -> List[SlideBarcode]:
        downsmple_factor = 0.1
        extracted_slide_barcodes = []
        while downsmple_factor <= 1:
            extracted_slide_barcodes = BarcodeExtractor._extract_slide_barcode(slide_path=slide_path, downsmple_factor=downsmple_factor, barcode_convention=barcode_convention)
            if len(extracted_slide_barcodes) > 0:
                break

            downsmple_factor *= 2
            if 0.5 < downsmple_factor < 1:
                downsmple_factor = 1

        return extracted_slide_barcodes

    def extract_barcodes(self):
        slide_file_paths = self._get_slide_paths()
        for slide_path in tqdm(slide_file_paths):
            extracted_slide_barcodes = BarcodeExtractor._extract_slide_barcode(slide_path=slide_path, barcode_convention=self._barcode_convention)

    def add_barcodes_to_slide_list(data_dir, dataset_name, scan_barcodes=True):
        barcode_convention = get_barcode_convention(data_dir)
        if barcode_convention == BARCODE_CONVENTION_NONE:
            return
        print('adding barcodes to slide list')

        start_time = time.time()
        prev_time = start_time

        barcode_list_file = get_barcode_list_file(data_dir, dataset_name)
        slide_list_df = dataset_utils.open_excel_file(barcode_list_file)
        barcode_list, comment_list, label_image_name_list = [], [], []

        for slide_info in tqdm(slide_list_df.iterrows()):
            # slide_ind = slide_info[0]
            print('extracting barcode from slide: ' + slide_info[1]['file'])
            barcode = []
            if scan_barcodes:
                barcode = extract_barcode_from_slide(slide_info, barcode_convention)

            label_image_name, comment = extract_slide_label(data_dir, slide_info, len(barcode))
            barcode = parse_barcode(barcode)

            barcode_list.append(barcode)
            comment_list.append(comment)
            label_image_name_list.append(label_image_name)

            image_time = time.time()
            print('processing time: ' + str(image_time - prev_time) + ' sec')
            prev_time = image_time

        save_barcode_list(slide_list_df, barcode_list, comment_list, barcode_list_file)
        write_label_images_to_excel(slide_list_df, label_image_name_list, data_dir, dataset_name)
        print('Finished, total time: ' + str(time.time() - start_time) + ' sec')
