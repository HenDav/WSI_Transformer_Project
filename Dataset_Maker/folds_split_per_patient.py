from Dataset_Maker import dataset_utils
import numpy as np
from random import shuffle
import pandas as pd
from dataclasses import dataclass


def split_dataset_into_folds(data_dir, dataset, fold_params, split_all_dataset_group=False):
    print('splitting the dataset into folds')
    if split_all_dataset_group:
        split_all_dataset_group_into_folds(fold_params, dataset)
    else:
        in_file = dataset_utils.get_slides_data_file(data_dir, dataset)
        dataset_utils.backup_dataset_metadata(in_file, extension='_before_folds')
        batch_num = dataset_utils.get_dataset_batch_num(dataset)
        single_batch_in_dataset = (batch_num == '') or (batch_num == 1)
        if single_batch_in_dataset:
            split_single_batch_into_folds(in_file, fold_params)
        else:
            split_single_dataset_into_folds_inline_w_dataset_group(in_file, fold_params)

    print('finished splitting the dataset')


def split_single_batch_into_folds(in_file, fold_params):
    slides_data = dataset_utils.open_excel_file(in_file)
    slides_data_folds = split_single_dataset_into_folds(slides_data, fold_params)
    dataset_utils.save_df_to_excel(slides_data_folds, in_file)


def split_single_dataset_into_folds(slides_data, fold_params):
    patients_set = set(slides_data[fold_params.patient_column_name])  # support mixture of strings and ints , to support "missing data" patients
    patients = np.array([patient for patient in patients_set], dtype='object')
    # note - all "Missing Data": patients will be in the same fold
    N_patients = patients.shape[0]

    fold_size = int(N_patients * (1 - fold_params.test_ratio - fold_params.val_ratio) / fold_params.n_folds)
    N_val = int(N_patients * fold_params.val_ratio)
    N_test = N_patients - N_val - fold_size * fold_params.n_folds

    folds = create_random_fold_list(N_test, N_val, fold_params, fold_size)
    slides_data_folds = add_folds_to_metadata(slides_data, patients, folds, fold_params)
    return slides_data_folds


def add_folds_to_metadata(slides_data, patients, folds, fold_params):
    if 'test fold idx' in slides_data.keys():
        slides_data['prev folds'] = slides_data['test fold idx']
        slides_data = slides_data.drop('test fold idx', 1)
    patients_folds_df = pd.DataFrame({fold_params.patient_column_name: patients, 'test fold idx': folds})
    slides_data_folds = slides_data.merge(right=patients_folds_df,
                                          left_on=fold_params.patient_column_name,
                                          right_on=fold_params.patient_column_name,
                                          how='outer')
    return slides_data_folds


def create_random_fold_list(N_test, N_val, fold_params, fold_size):
    folds = ['test'] * N_test

    if fold_params.test_ratio == 0:  # replace test values with random folds
        folds = list(np.random.randint(1, fold_params.n_folds + 1, size=N_test))

    folds.extend(['val'] * N_val)

    if fold_params.n_folds == 1:
        folds.extend(['train'] * fold_size)
    else:
        for ii in np.arange(1, fold_params.n_folds + 1):
            folds.extend(list(np.ones(fold_size, dtype=int) * ii))
    shuffle(folds)
    return folds


def split_all_dataset_group_into_folds(fold_params, dataset):
    dataset_group = dataset_utils.get_dataset_group(dataset)
    dataset_utils.backup_all_dataset_group_metadata(dataset_group, extension='_before_folds')
    slides_data = dataset_utils.merge_dataset_group_metadata(dataset_group)
    slides_data_folds = split_single_dataset_into_folds(slides_data, fold_params)
    dataset_utils.unmerge_dataset_group_data(slides_data_folds, dataset_group)


def split_single_dataset_into_folds_inline_w_dataset_group(in_file, fold_params):
    pass #todo


@dataclass
class fold_split_params:
    n_folds: int = 5
    test_ratio: float = 0.25
    val_ratio: float = 0
    patient_column_name: str = 'patient barcode'

