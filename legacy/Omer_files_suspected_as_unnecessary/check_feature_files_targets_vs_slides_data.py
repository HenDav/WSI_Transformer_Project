import Dataset_Maker.dataset_utils
import utils
import argparse
import datasets
from torch.utils.data import DataLoader
import pandas as pd
import os
import sys

import utils_MIL

parser = argparse.ArgumentParser(description='Check traget inconsistency')
parser.add_argument('-tf', '--test_fold', default=1, type=int, help='fold to be as TEST FOLD')
parser.add_argument('-tar', '--target', default='Her2', type=str, help='label: Her2/ER/PR/EGFR/PDL1')
parser.add_argument('-ds', '--dataset', type=str, default='CAT', help='DataSet to use')
args = parser.parse_args()

if sys.platform == 'darwin':
    args.target = 'ER'

# Open Slide_data
print('Loading slides_data')
dir_dict = Dataset_Maker.dataset_utils.get_datasets_dir_dict(Dataset=args.dataset)
for _, key in enumerate(dir_dict):
    slide_meta_data_file = os.path.join(dir_dict[key], 'slides_data_' + key + '.xlsx')
    if 'meta_data_DF' not in locals():
        meta_data_DF = pd.read_excel(slide_meta_data_file)
    else:
        meta_data_DF = meta_data_DF.append(pd.read_excel(slide_meta_data_file))

# Open feature files
print('Opening feature files')
data_location = utils_MIL.get_RegModel_Features_location_dict(train_DataSet=args.dataset, target=args.target, test_fold=args.test_fold)

train_dset = datasets.Features_MILdataset(dataset=args.dataset,
                                          data_location=data_location['TrainSet Location'],
                                          bag_size=1,
                                          target=args.target,
                                          is_train=True,
                                          test_fold=args.test_fold
                                          )

test_dset = datasets.Features_MILdataset(dataset=args.dataset,
                                         data_location=data_location['TestSet Location'],
                                         bag_size=1,
                                         target=args.target,
                                         is_train=False,
                                         test_fold=args.test_fold
                                         )

train_loader = DataLoader(train_dset, batch_size=50, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dset, batch_size=50, shuffle=False, num_workers=0, pin_memory=True)

loader_dict = {'Train': train_loader,
               'Test': test_loader}
print('Comparing slides_data.xlsx to feature files')
for loader_key in loader_dict.keys():
    slide_names, slide_targets = [], []
    bad_slides, all_slides = 0, 0
    for idx, minibatch in enumerate(loader_dict[loader_key]):
        slide_names.extend(minibatch['slide name'])
        slide_targets.extend(minibatch['targets'])


    # Going over all slides and checking the targets:
    for idx, slide in enumerate(slide_names):
        try:
            target_from_slides_data = meta_data_DF.loc[meta_data_DF['file'] == slide][args.target + ' status'].values[0]
        except IndexError:
            print(slide)
        #print(target_from_slides_data)
        if target_from_slides_data == 'Positive':
            target_true = 1
        elif target_from_slides_data == 'Negative':
            target_true = 0
        else:
            target_true = -1

        target_from_file = slide_targets[idx]

        all_slides += 1
        if target_from_file != target_true:
            bad_slides += 1
            print('{} Set Mismatch in slide {}. Target is {}, Feature target is {}'.format(loader_key,
                                                                                           slide,
                                                                                           target_from_slides_data,
                                                                                           target_from_file
                                                                                           ))



    print('Total of {}/{} bad slides in {} set'.format(bad_slides, all_slides, loader_key))
print('Done')