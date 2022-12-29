import pandas as pd
import argparse

import Dataset_Maker.dataset_utils
import utils
import os
import numpy as np

parser = argparse.ArgumentParser(description='Check traget inconsistency')
parser.add_argument('-tf', '--test_fold', default=2, type=int, help='fold to be as TEST FOLD')
parser.add_argument('-ds', '--dataset', type=str, default='CAT', help='DataSet to use')
parser.add_argument('-tar', '--target', default='Her2', type=str, help='label: Her2/ER/PR/EGFR/PDL1')
parser.add_argument('--is_train', action='store_true', help='check trainset')
args = parser.parse_args()

# Get locations:
dir_dict = Dataset_Maker.dataset_utils.get_datasets_dir_dict(Dataset=args.dataset)
all_slides, all_patients, all_inconsistent_slides, all_inconsistent_patients = 0, 0, 0, 0

for _, key in enumerate(dir_dict):
    inconsistent_slides, inconsistent_patients, total_slides, total_patients = 0, 0, 0, 0
    patient_dict = {}

    slide_meta_data_file = os.path.join(dir_dict[key], 'slides_data_' + key + '.xlsx')
    slide_meta_data_DF = pd.read_excel(slide_meta_data_file)
    print(key, len(slide_meta_data_DF))

    if args.dataset == 'CAT' or args.dataset == 'ABCTB_TCGA':
        fold_column_name = 'test fold idx breast'
    else:
        fold_column_name = 'test fold idx'

    if args.is_train:
        folds = list(slide_meta_data_DF[fold_column_name].unique())
        folds.remove(args.test_fold)
        if 'test' in folds:
            folds.remove('test')
        if 'val' in folds:
            folds.remove('val')
    else:
        folds = [args.test_fold]
        folds.append('val')

    correct_folds = slide_meta_data_DF[fold_column_name].isin(folds)
    valid_slide_indices = np.array(correct_folds.index[correct_folds])

    slide_names = slide_meta_data_DF['file'][valid_slide_indices].to_list()
    targets = slide_meta_data_DF[args.target + ' status'][valid_slide_indices].to_list()
    patients = slide_meta_data_DF['patient barcode'][valid_slide_indices].to_list()

    # Gathering patient data in a dict.
    for idx, patient in enumerate(patients):
        if patient not in patient_dict.keys():
            patient_dict[patient] = {'Slides': [slide_names[idx]],
                                     'Targets': [targets[idx]]
                                     }
        else:
            patient_dict[patient]['Slides'].append(slide_names[idx])
            patient_dict[patient]['Targets'].append(targets[idx])

    # For each dataset we'll check target inconsistency:


    for patient in patient_dict.keys():
        targets = patient_dict[patient]['Targets']
        total_patients += 1
        total_slides += len(targets)
        if targets.count(targets[0]) != len(targets):
            inconsistent_patients += 1
            inconsistent_slides += len(targets)

    all_slides += total_slides
    all_patients += total_patients
    all_inconsistent_slides += inconsistent_slides
    all_inconsistent_patients += inconsistent_patients

    print('in {} dataset {}, there are {}/{} inconsistent slides and {}/{} inconsistent patients'.format('Train' if args.is_train else 'Test',
                                                                                                         key,
                                                                                                         inconsistent_slides,
                                                                                                         total_slides,
                                                                                                         inconsistent_patients,
                                                                                                         total_patients))




print('in {} dataset {} with TestFold {}, there are {}/{} inconsistent slides and {}/{} inconsistent patients'.format('Train' if args.is_train else 'Test',
                                                                                                                      args.dataset,
                                                                                                                      args.test_fold,
                                                                                                                      all_inconsistent_slides,
                                                                                                                      all_slides,
                                                                                                                      all_inconsistent_patients,
                                                                                                                      all_patients))





