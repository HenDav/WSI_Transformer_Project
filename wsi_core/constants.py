# python peripherals
import os
from typing import Dict, List, Tuple, Set
from pathlib import Path

# General parameters
test_fold_id = 'test'
max_attempts = 10
white_ratio_threshold = 0.5
white_intensity_threshold = 170
current_mpp = 1.0 # TODO: maybe infer it directly from the folder?
main_metadata_csv = '/home/dahen/WSI_ran_legacy/royve/WSI/metadata.csv' # TODO: change to proper central metadata once Carmel9-11 & Haemek are incorporated.
data_root_gipdeep10 = '/data/unsynced_data/h5'
data_root_netapp = ''

# Invalid values
invalid_values = ['Missing Data', 'Not performed', '[Not Evaluated]', '[Not Available]']
invalid_value = 'NA'
invalid_fold_column_names = ['test fold idx breast', 'test fold idx', 'test fold idx breast - original for carmel']
slides_with_invalid_values = ['TCGA-OL-A5S0-01Z-00-DX1.49A7AC9D-C186-406C-BA67-2D73DE82E13B.svs']

# Dataset ids
dataset_id_cat = 'CAT'
dataset_id_ta = 'TA'
dataset_id_abctb = 'ABCTB'
dataset_id_sheba = 'SHEBA'
dataset_id_tcga = 'TCGA'
dataset_id_carmel = 'CARMEL'
number_of_carmel_train_batches = 8
dataset_containment_dict = {dataset_id_cat: [dataset_id_ta, dataset_id_carmel], 
                            dataset_id_ta: [dataset_id_abctb, dataset_id_tcga], 
                            dataset_id_carmel: [f"{dataset_id_carmel}{batch_num}" for batch_num in range(1, number_of_carmel_train_batches)]}
folds_for_datasets = {f"{dataset_id_carmel}{batch_num}": [1, 2, 3, 4, 5] for batch_num in range(1, number_of_carmel_train_batches)}
folds_for_datasets[dataset_id_tcga] = [1, 2, 3, 4, 5]
folds_for_datasets[dataset_id_abctb] = [1, 2, 3, 4, 5]
metadata_base_dataset_ids = [dataset_id_abctb, dataset_id_sheba, dataset_id_carmel, dataset_id_tcga]

# Grid data
bad_segmentation_column_name = 'bad segmentation'
grids_data_prefix = 'slides_data_'
grid_data_file_name = 'Grid_data.xlsx'

# Carmel
slide_barcode_column_name_carmel = 'slide barcode'
slide_barcode_column_name_enhancement_carmel = 'TissueID'
patient_barcode_column_name_enhancement_carmel = 'PatientIndex'
block_id_column_name_enhancement_carmel = 'BlockID'

# TCGA
patient_barcode_column_name_enhancement_tcga = 'Sample CLID'
slide_barcode_prefix_column_name_enhancement_tcga = 'Sample CLID'

# ABCTB
file_column_name_enhancement_abctb = 'Image File'
patient_barcode_column_name_enhancement_abctb = 'Identifier'

# SHEBA
er_status_column_name_sheba = 'ER '
pr_status_column_name_sheba = 'PR '
her2_status_column_name_sheba = 'HER-2 IHC '
grade_column_name_sheba = 'Grade'
tumor_type_column_name_sheba = 'Histology'

# Shared
file_column_name_shared = 'file'
patient_barcode_column_name_shared = 'patient barcode'
dataset_id_column_name_shared = 'dataset name'
mpp_column_name_shared = 'MPP'
scan_date_column_name_shared = 'Scan Date'
width_column_name_shared = 'Width'
height_column_name_shared = 'Height'
magnification_column_name_shared = 'Manipulated Objective Power'
er_status_column_name_shared = 'ER status'
pr_status_column_name_shared = 'PR status'
her2_status_column_name_shared = 'Her2 status'
fold_column_name_shared = 'test fold idx'

# Curated
file_column_name = 'file'
patient_barcode_column_name = 'patient_barcode'
dataset_id_column_name = 'id'
mpp_column_name = 'mpp'
scan_date_column_name = 'scan_date'
width_column_name = 'width'
height_column_name = 'height'
magnification_column_name = 'magnification'
er_status_column_name = 'er_status'
pr_status_column_name = 'pr_status'
her2_status_column_name = 'her2_status'
fold_column_name = 'fold'
grade_column_name = 'grade'
tumor_type_column_name = 'tumor_type'
slide_barcode_column_name = 'slide_barcode'
slide_barcode_prefix_column_name = 'slide_barcode_prefix'
legitimate_tiles_column_name = 'legitimate_tiles'
total_tiles_column_name = 'total_tiles'
tile_usage_column_name = 'tile_usage'


def get_path_suffixes() -> Dict[str, Path]:
    path_suffixes = {
        dataset_id_tcga: f'Breast/{dataset_id_tcga}',
        dataset_id_abctb: f'Breast/{dataset_id_abctb}_TIF',
    }

    for i in range(1, 12):
        if i in range(1, 9):
            batches = '1-8'
        else:
            batches = '9-11'
        path_suffixes[f'{dataset_id_carmel}{i}'] = Path(f'Breast/{dataset_id_carmel.capitalize()}/{batches}/Batch_{i}/{dataset_id_carmel}{i}')

    for i in range(2, 7):
        path_suffixes[f'{dataset_id_sheba}{i}'] = Path(f'Breast/{dataset_id_sheba.capitalize()}/Batch_{i}/{dataset_id_sheba}{i}')

    return path_suffixes


def get_dataset_paths(datasets_base_dir_path: Path) -> Dict[str, Path]:
    dataset_paths = {}
    path_suffixes = get_path_suffixes()

    for k in path_suffixes.keys():
        path_suffix = path_suffixes[k]
        dataset_paths[k] = Path(os.path.normpath(os.path.join(datasets_base_dir_path, path_suffix)))

    return dataset_paths

def get_dataset_ids(dataset_id: str) -> Set[str]:
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
            set_contains_dict_keys = set_contains_dict_keys or dataset_id in dataset_containment_dict.keys()
    
    return dataset_ids