import os
from pathlib import Path
from typing import Dict, List
from itertools import takewhile
import re

import numpy
import pandas

# General parameters
test_fold_id = "test"
max_attempts = 10
white_ratio_threshold = 0.5
white_intensity_threshold = 170
current_mpp = 1.0  # TODO: maybe infer it directly from the folder?
main_metadata_csv = "/home/dahen/WSI_ran_legacy/royve/WSI/metadata.csv"  # TODO: change to proper central metadata once Carmel9-11 & Haemek are incorporated.
data_root_gipdeep10 = "/data/unsynced_data/h5"
data_root_netapp = ""

# Invalid values
invalid_values = ["Missing Data", "Not performed", "[Not Evaluated]", "[Not Available]", "Was not stained", numpy.nan]
invalid_value = pandas.NA
invalid_fold_column_names = [
    "test fold idx breast",
    "test fold idx",
    "test fold idx breast - original for carmel",
]
slides_with_invalid_values = [
    "TCGA-OL-A5S0-01Z-00-DX1.49A7AC9D-C186-406C-BA67-2D73DE82E13B.svs"
]

# Dataset ids
dataset_id_cat = "CAT"
dataset_id_ta = "TA"
dataset_id_abctb = "ABCTB"
dataset_id_sheba = "SHEBA"
dataset_id_tcga = "TCGA"
dataset_id_carmel = "CARMEL"
dataset_id_haemek = "HAEMEK"
number_of_carmel_train_batches = 8
dataset_containment_dict = {
    dataset_id_cat: (dataset_id_ta, dataset_id_carmel),
    dataset_id_ta: (dataset_id_abctb, dataset_id_tcga),
    dataset_id_carmel: tuple(
        [
            f"{dataset_id_carmel}{batch_num}"
            for batch_num in range(1, number_of_carmel_train_batches + 1)
        ]
    ),
}
folds_for_datasets = {
    f"{dataset_id_carmel}{batch_num}": (1, 2, 3, 4, 5)
    for batch_num in range(1, number_of_carmel_train_batches + 1)
}
folds_for_datasets[dataset_id_tcga] = (1, 2, 3, 4, 5)
folds_for_datasets[dataset_id_abctb] = (1, 2, 3, 4, 5)
metadata_base_dataset_ids = (
    dataset_id_abctb,
    dataset_id_sheba,
    dataset_id_carmel,
    dataset_id_tcga,
    dataset_id_haemek
)

# Grid data
bad_segmentation_column_name = "bad segmentation"
grids_data_prefix = "slides_data_"
grid_data_file_name = "Grid_data.xlsx"

# Haemek
file_column_name_haemek = "file"
patient_barcode_column_name_haemek = "patient barcode"
dataset_id_column_name_haemek = "id"
mpp_column_name_haemek = "MPP"
scan_date_column_name_haemek = "Scan Date"
width_column_name_haemek = "Width"
height_column_name_haemek = "Height"
magnification_column_name_haemek = "Manipulated Objective Power"
er_status_column_name_haemek = "ER status"
pr_status_column_name_haemek = "PR status"
her2_status_column_name_haemek = "Her2 status"
ki_67_status_column_name_haemek = "Ki67 status"
fold_column_name_haemek = "test fold idx"
tumor_type_column_name_haemek = "TumorType"

# Carmel
file_column_name_carmel = "file"
patient_barcode_column_name_carmel = "patient barcode"
dataset_id_column_name_carmel = "id"
mpp_column_name_carmel = "MPP"
scan_date_column_name_carmel = "Scan Date"
width_column_name_carmel = "Width"
height_column_name_carmel = "Height"
magnification_column_name_carmel = "Manipulated Objective Power"
er_status_column_name_carmel = "ER status"
pr_status_column_name_carmel = "PR status"
her2_status_column_name_carmel = "Her2 status"
ki_67_status_column_name_carmel = "Ki67 status"
fold_column_name_carmel = "test fold idx"
slide_barcode_column_name_carmel = "slide barcode"
slide_barcode_column_name_enhancement_carmel = "TissueID"
patient_barcode_column_name_enhancement_carmel = "PatientIndex"
block_id_column_name_enhancement_carmel = "BlockID"

# TCGA
file_column_name_tcga = "file"
patient_barcode_column_name_tcga = "patient barcode"
dataset_id_column_name_tcga = "dataset name"
mpp_column_name_tcga = "MPP"
scan_date_column_name_tcga = "Scan Date"
width_column_name_tcga = "Width"
height_column_name_tcga = "Height"
magnification_column_name_tcga = "Manipulated Objective Power"
er_status_column_name_tcga = "ER status"
pr_status_column_name_tcga = "PR status"
her2_status_column_name_tcga = "Her2 status"
fold_column_name_tcga = "test fold idx"
patient_barcode_column_name_enhancement_tcga = "Sample CLID"
slide_barcode_prefix_column_name_enhancement_tcga = "Sample CLID"

# ABCTB
file_column_name_abctb = "file"
patient_barcode_column_name_abctb = "patient barcode"
dataset_id_column_name_abctb = "dataset name"
mpp_column_name_abctb = "MPP"
scan_date_column_name_abctb = "Scan Date"
width_column_name_abctb = "Width"
height_column_name_abctb = "Height"
magnification_column_name_abctb = "Manipulated Objective Power"
er_status_column_name_abctb = "ER status"
pr_status_column_name_abctb = "PR status"
her2_status_column_name_abctb = "Her2 status"
fold_column_name_abctb = "test fold idx"
file_column_name_enhancement_abctb = "Image File"
patient_barcode_column_name_enhancement_abctb = "Identifier"

# SHEBA
file_column_name_sheba = "file"
patient_barcode_column_name_sheba = "patient barcode"
dataset_id_column_name_sheba = "dataset name"
mpp_column_name_sheba = "MPP"
scan_date_column_name_sheba = "Scan Date"
width_column_name_sheba = "Width"
height_column_name_sheba = "Height"
magnification_column_name_sheba = "Manipulated Objective Power"
fold_column_name_sheba = "test fold idx"
er_status_column_name_sheba = "ER "
pr_status_column_name_sheba = "PR "
her2_status_column_name_sheba = "HER-2 IHC "
grade_column_name_sheba = "Grade"
tumor_type_column_name_sheba = "Histology"
onco_ki_67_column_name_sheba = "Proliferation (Ki-67) Oncotype"
onco_score_11_column_name_sheba = "onco_score_11 status"
onco_score_18_column_name_sheba = "onco_score_18 status"
onco_score_26_column_name_sheba = "onco_score_26 status"
onco_score_31_column_name_sheba = "onco_score_31 status"
onco_score_all_column_name_sheba = "onco_score_all status"

# Shared
# file_column_name_shared = "file"
# patient_barcode_column_name_shared = "patient barcode"
# dataset_id_column_name_shared = "dataset name"
# mpp_column_name_shared = "MPP"
# scan_date_column_name_shared = "Scan Date"
# width_column_name_shared = "Width"
# height_column_name_shared = "Height"
# magnification_column_name_shared = "Manipulated Objective Power"
# er_status_column_name_shared = "ER status"
# pr_status_column_name_shared = "PR status"
# her2_status_column_name_shared = "Her2 status"
# fold_column_name_shared = "test fold idx"

# Curated
file_column_name = "file"
patient_barcode_column_name = "patient_barcode"
dataset_id_column_name = "id"
mpp_column_name = "mpp"
scan_date_column_name = "scan_date"
width_column_name = "width"
height_column_name = "height"
magnification_column_name = "magnification"
er_status_column_name = "er_status"
pr_status_column_name = "pr_status"
her2_status_column_name = "her2_status"
fold_column_name = "fold"
grade_column_name = "grade"
tumor_type_column_name = "tumor_type"
slide_barcode_column_name = "slide_barcode"
slide_barcode_prefix_column_name = "slide_barcode_prefix"
legitimate_tiles_column_name = "legitimate_tiles"
tiles_count_column_name = "tiles_count"
total_tiles_column_name = "total_tiles"
tile_usage_column_name = "tile_usage"
ki_67_status_column_name = "ki_67_status"
onco_ki_67_column_name = "onco_ki_67"
onco_score_11_column_name = "onco_score_11"
onco_score_18_column_name = "onco_score_18"
onco_score_26_column_name = "onco_score_26"
onco_score_31_column_name = "onco_score_31"
onco_score_all_column_name = "onco_score_all"


def get_path_suffixes() -> Dict[str, Path]:
    path_suffixes = {
        dataset_id_tcga: f"Breast/{dataset_id_tcga}",
        dataset_id_abctb: f"Breast/{dataset_id_abctb}_TIF",
    }

    for i in range(1, 12):
        if i in range(1, 9):
            batches = "1-8"
        else:
            batches = "9-11"
        path_suffixes[f"{dataset_id_carmel}{i}"] = Path(
            f"Breast/{dataset_id_carmel.capitalize()}/{batches}/Batch_{i}/{dataset_id_carmel}{i}"
        )

    for i in range(2, 7):
        path_suffixes[f"{dataset_id_sheba}{i}"] = Path(
            f"Breast/{dataset_id_sheba.capitalize()}/Batch_{i}/{dataset_id_sheba}{i}"
        )

    for i in range(1, 4):
        path_suffixes[f"{dataset_id_haemek}{i}"] = Path(
            f"Breast/{dataset_id_haemek.capitalize()}/Batch_{i}/{dataset_id_haemek}{i}"
        )

    return path_suffixes


def get_dataset_paths(datasets_base_dir_path: Path) -> Dict[str, Path]:
    dataset_paths = {}
    path_suffixes = get_path_suffixes()

    for k in path_suffixes.keys():
        path_suffix = path_suffixes[k]
        dataset_paths[k] = Path(
            os.path.normpath(os.path.join(datasets_base_dir_path, path_suffix))
        )

    return dataset_paths


def get_dataset_ids(dataset_id: str) -> List[str]:
    dataset_ids = set([dataset_id])
    dataset_ids_updated = set()
    set_contains_dict_keys = dataset_id in dataset_containment_dict.keys()

    while set_contains_dict_keys:
        for dataset_id in dataset_ids:
            if dataset_id in dataset_containment_dict.keys():
                for contained_dataset_id in dataset_containment_dict[dataset_id]:
                    dataset_ids_updated.add(contained_dataset_id)
            else:
                dataset_ids_updated.add(dataset_id)
        dataset_ids = dataset_ids_updated.copy()
        dataset_ids_updated = set()
        set_contains_dict_keys = False
        for dataset_id in dataset_ids:
            set_contains_dict_keys = (
                set_contains_dict_keys or dataset_id in dataset_containment_dict.keys()
            )

    return sorted(list(dataset_ids))


def get_dataset_id_suffix(dataset_id: str) -> int:
    return int(re.findall('\d+', dataset_id)[0])
