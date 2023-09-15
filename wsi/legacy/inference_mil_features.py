import argparse
import os
import sys

import matplotlib.pyplot as plt
from Nets import nets_mil
import numpy as np
import torch
import torch.nn as nn
from cycler import cycler
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import utils
import utils_MIL

parser = argparse.ArgumentParser(
    description='WSI_MIL Features Slide inference')
parser.add_argument('-ex',
                    '--experiment',
                    type=int,
                    default=10679,
                    help='Use this model for inference')
parser.add_argument('-fe',
                    '--from_epoch',
                    type=int,
                    default=[500],
                    help='Use this epoch model for inference')
parser.add_argument('-pp',
                    '--is_per_patient',
                    action='store_true',
                    help='per patient inference ?')
parser.add_argument('-conly',
                    '--carmel_only',
                    action='store_true',
                    help='use slides only from Carmel Dataset ?')
parser.add_argument('-sts',
                    '--save_tile_scores',
                    dest='save_tile_scores',
                    action='store_true',
                    help='save tile scores')
parser.add_argument('-cts',
                    '--carmel_test_set',
                    dest='carmel_test_set',
                    action='store_true',
                    help='run inference over carmel batch 9-11 ?')
parser.add_argument('--batch',
                    type=int,
                    default=9,
                    help='Batch No. for carmel test dataset (9, 10, 11)')
parser.add_argument('--haemek_test_set',
                    dest='haemek_test_set',
                    action='store_true',
                    help='run inference over HAEMEK ?')
parser.add_argument('--TA_test_set',
                    dest='TA_test_set',
                    action='store_true',
                    help='run inference over TA fold 6 ?')
parser.add_argument('--hic_test_set',
                    dest='hic_test_set',
                    action='store_true',
                    help='run inference over hic ?')
parser.add_argument('--SHEBA_test_set',
                    dest='SHEBA_test_set',
                    action='store_true',
                    help='run inference over SHEBA fold 6 ?')
parser.add_argument('--PORTUGAL_test_set',
                    dest='PORTUGAL_test_set',
                    action='store_true',
                    help='run inference over portugal ?')
#parser.add_argument('-nt', '--num_tiles', type=int, default=500, help='Number of tiles to use')
#parser.add_argument('-ds', '--dataset', type=str, default='HEROHE', help='DataSet to use')
#parser.add_argument('-f', '--folds', type=list, default=[2], help=' folds to infer')
#parser.add_argument('--model', default='resnet50_gn', type=str, help='resnet50_gn / receptornet') # RanS 15.12.20
args = parser.parse_args()

EPS = 1e-7
if type(args.from_epoch) == int:
    args.from_epoch = [args.from_epoch]

#args.carmel_test_set = False
#args.haemek_test_set = True

custom_cycler = (cycler(color=[
    '#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3',
    '#999999', '#e41a1c', '#dede00'
]) + cycler(linestyle=[
    'solid', 'dashed', 'dotted', 'dashdot', 'solid', 'dashed', 'dotted',
    'dashdot', 'dashed'
]))

# Device definition:
DEVICE = utils.device_gpu_cpu()

# Get number of available CPUs:
cpu_available = utils.get_cpu()

# Data type definition:
DATA_TYPE = 'Features'

# Loss criterion definition:
criterion = nn.CrossEntropyLoss()

# Load saved model:
print('Loading pre-saved model from Exp. {} and epoch {}'.format(
    args.experiment, args.from_epoch))
run_data_output = utils.run_data(experiment=args.experiment)
output_dir, test_fold, dataset, target, model_name, free_bias, CAT_only, is_tumor_train_mode =\
    run_data_output['Location'], run_data_output['Test Fold'], run_data_output['Dataset Name'], run_data_output['Receptor'],\
    run_data_output['Model Name'], run_data_output['Free Bias'], run_data_output['CAT Only'], run_data_output['Receptor + is_Tumor Train Mode']

if 'self-supervised-pathology' in output_dir:
    output_dir = output_dir.replace('self-supervised-pathology', 'wsi/legacy')

if sys.platform == 'darwin':
    # fix output_dir:
    if output_dir.split('/')[1] == 'home':
        output_dir = '/'.join(output_dir.split('/')[-2:])

CAT_dsets = [
    r'FEATURES: Exp_355-ER-TestFold_1', r'FEATURES: Exp_393-ER-TestFold_2',
    r'FEATURES: Exp_472-ER-TestFold_3', r'FEATURES: Exp_542-ER-TestFold_4',
    r'FEATURES: Exp_611-ER-TestFold_5', r'FEATURES: Exp_392-Her2-TestFold_1',
    r'FEATURES: Exp_412-Her2-TestFold_2',
    r'FEATURES: Exp_20114-Her2-TestFold_3',
    r'FEATURES: Exp_20201-Her2-TestFold_4',
    r'FEATURES: Exp_20228-Her2-TestFold_5', r'FEATURES: Exp_10-PR-TestFold_1',
    r'FEATURES: Exp_20063-PR-TestFold_2', r'FEATURES: Exp_497-PR-TestFold_3',
    r'FEATURES: Exp_20207-PR-TestFold_4', r'FEATURES: Exp_20235-PR-TestFold_5',
    r'FEATURES: Exp_355-ER-TestFold_-1+Exp_20010-PR-TestFold_-1'
]

CAT_with_Location_dsets = [
    r'FEATURES: Exp_392-Her2-TestFold_1 With Locations',
    r'FEATURES: Exp_392-Her2-TestFold_1 With Locations + IS_TUMOR',
    r'FEATURES: Exp_10-PR-TestFold_1  With Locations + IS_TUMOR',
    r'FEATURES: Exp_355-ER-TestFold_1 With Locations for is_Tumor + IS_TUMOR',
    r'FEATURES: Exp_355-ER-TestFold_1 With Locations for is_Tumor',
    r'FEATURES: Exp_355-ER-TestFold_1 With Locations',
    r'FEATURES: Exp_10-PR-TestFold_1  With Locations',
    r'FEATURES: Exp_392-Her2-TestFold_1 With Locations',
    r'FEATURES: Exp_393-ER-TestFold_2 With Locations',
    r'FEATURES: Exp_412-Her2-TestFold_2 With Locations',
    r'FEATURES: Exp_20063-PR-TestFold_2 With Locations'
]

CARMEL_dsets = [
    r'FEATURES: Exp_419-Ki67-TestFold_1', r'FEATURES: Exp_490-Ki67-TestFold_2',
    r'FEATURES: Exp_20152-Ki67-TestFold_3', r'FEATURES: Exp_30090-Ki67-TestFold_4', r'FEATURES: Exp_30092-Ki67-TestFold_5'
]
ABCTB_dsets = [r'FEATURES: Exp_20094-survival-TestFold_1']
TCGA_ABCTB_dsets = [
    r'FEATURES: Exp_293-ER-TestFold_1', r'FEATURES: Exp_309-PR-TestFold_1',
    r'FEATURES: Exp_308-Her2-TestFold_1', r'FEATURES: Exp_299-ER-TestFold_2',
    r'FEATURES: Exp_310-Her2-TestFold_2', r'FEATURES: Exp_320-PR-TestFold_2',
    r'FEATURES: Exp_301-ER-TestFold_3', r'FEATURES: Exp_311-Her2-TestFold_3',
    r'FEATURES: Exp_312-PR-TestFold_3', r'FEATURES: Exp_302-ER-TestFold_4',
    r'FEATURES: Exp_334-Her2-TestFold_4', r'FEATURES: Exp_313-PR-TestFold_4',
    r'FEATURES: Exp_303-ER-TestFold_5', r'FEATURES: Exp_348-Her2-TestFold_5',
    r'FEATURES: Exp_321-PR-TestFold_5'
]

HAEMEK_transfer_dsets = [
    r'FEATURES: Exp_50015-ER-TestFold_1 transfered from CAT',
    r'FEATURES: Exp_50143-ER-TestFold_2 transfered from CAT',
    r'FEATURES: Exp_50157-ER-TestFold_3 transfered from CAT',
    r'FEATURES: Exp_50171-ER-TestFold_4 transfered from CAT'
]

HAEMEK_finetuned_dsets = [
    r'FEATURES: Exp_50151-ER-TestFold_1 finetuned from CAT',
    r'FEATURES: Exp_50161-ER-TestFold_2 finetuned from CAT',
    r'FEATURES: Exp_50163-ER-TestFold_3 finetuned from CAT',
    r'FEATURES: Exp_50172-ER-TestFold_4 finetuned from CAT'
]

HAEMEK_dsets = [
    r'FEATURES: Exp_50014-ER-TestFold_1 trained from scratch',
    r'FEATURES: Exp_50142-ER-TestFold_2 trained from scratch',
    r'FEATURES: Exp_50159-ER-TestFold_3 trained from scratch',
    r'FEATURES: Exp_50175-ER-TestFold_4 trained from scratch'
]

SHEBA_dsets = [
    r'FEATURES: Exp_50197-onco_score_26-TestFold_1',
    r'FEATURES: Exp_50200-onco_score_26-TestFold_2',
    r'FEATURES: Exp_50202-onco_score_26-TestFold_3',
    r'FEATURES: Exp_50203-onco_score_26-TestFold_4',
    r'FEATURES: Exp_50204-onco_score_26-TestFold_5',
    r'FEATURES: Exp_50209-onco_score_26-TestFold_-1',
    r'FEATURES: Exp_50212-onco_score_31-TestFold_-1'
]

if run_data_output['Dataset Name'] in CAT_dsets:
    dset = 'CAT'
elif run_data_output['Dataset Name'] in CAT_with_Location_dsets:
    dset = 'CAT with Location'
elif run_data_output['Dataset Name'] in CARMEL_dsets:
    dset = 'CARMEL'
elif run_data_output['Dataset Name'] in ABCTB_dsets:
    dset = 'ABCTB'
elif run_data_output['Dataset Name'] in TCGA_ABCTB_dsets:
    dset = 'TCGA_ABCTB'
elif run_data_output['Dataset Name'] in HAEMEK_finetuned_dsets:
    dset = 'HAEMEK_finetuned'
elif run_data_output['Dataset Name'] in HAEMEK_transfer_dsets:
    dset = 'HAEMEK_transfer'
elif run_data_output['Dataset Name'] in HAEMEK_dsets:
    dset = 'HAEMEK->HAEMEK'
elif run_data_output['Dataset Name'] in SHEBA_dsets:
    dset = 'SHEBA'
else:
    dset = None

    # Default is using features extracted with CAT trained extractor. Any other extractor used warrents the dset name
    # extractor_dataset->test_dataset
    
if args.carmel_test_set:
    if args.experiment in [30086, 30087, 30089] + list(range(30093, 30105)) + [40072]:  # [10586, 10587, 10590]:
        dset = 'TCGA_ABCTB->CARMEL'
    elif args.experiment in [40068,40067, 50218]:
        dset = 'CARMEL->CARMEL 9-11'
    else:
        dset = 'CARMEL 9-11' # Implicitly CAT->CARMEL 9-11

if args.haemek_test_set:
    if args.experiment in [30086, 30087, 30089] + list(range(30093, 30105)) + [40072]:
        dset = 'TCGA_ABCTB->HAEMEK'
    elif args.experiment in [40068,40067]:
        dset = 'CARMEL->HAEMEK'
    else:
        dset = 'HAEMEK' # Implicitly CAT->HAEMEK

if args.TA_test_set:
    if args.experiment in list(range(30065,30070)) + list(range(30071,30074)) + list(range(30075, 30082)) + [40064, 40074, 40066]:
        dset = 'CAT->TA 6'
    elif args.experiment in [30086, 30087, 30089] + list(range(30093,30105)) + [40072, 50225, 50226]:
        dset = 'TCGA_ABCTB->TA 6'
    elif args.experiment == 40067:
        dset = 'CARMEL->TA 6'
        
if args.PORTUGAL_test_set:
    if args.experiment in list(range(30065,30070)) + list(range(30071,30074)) + list(range(30075, 30082)) + [40064, 40074, 40066]:
        dset = 'CAT->PORTUGAL'
    elif args.experiment in [30086, 30087, 30089] + list(range(30093,30105)) + [40072, 50225, 50226]:
        dset = 'TCGA_ABCTB->PORTUGAL'
    elif args.experiment == 40067:
        dset = 'CARMEL->PORTUGAL'

if args.SHEBA_test_set:
    if args.experiment in [50199, 50201, 50205, 50206, 50208, 50211, 50213]:
        dset = 'SHEBA->SHEBA 6'
    
if args.hic_test_set:
    if args.experiment in [30086, 30087, 30089] + list(range(30093, 30105)) + [40072]:
        dset = 'TCGA_ABCTB->hic'
    elif args.experiment in [40068,40067]:
        dset = 'CARMEL->hic'
    else:
        dset = 'HIC' # Implicitly CAT->HIC

if dset == None:
    raise Exception('Dataset must be chosen')

data_4_inference = utils_MIL.get_RegModel_Features_location_dict(
    train_DataSet=dset,
    target=run_data_output['Receptor'].replace(
        '_Features', '', 1),  #run_data_output['Receptor'].split('_')[0],
    test_fold=run_data_output['Test Fold'])
if '+ IS_TUMOR' in run_data_output['Dataset Name']:
    args.target = target.replace('_Features', '', 1)
    test_data_dir = (data_4_inference[0]['TestSet Location'],
                     data_4_inference[1]['TestSet Location'])
else:
    test_data_dir = data_4_inference['TestSet Location']
'''
args.save_tile_scores = True
is_per_patient = False
#is_per_patient = False if args.save_tile_scores else True
carmel_only = False
'''
if args.haemek_test_set:
    dset = 'HAEMEK'
if args.hic_test_set:
    dset = 'HIC'
if args.carmel_test_set:
    if dset == 'CARMEL 9-11':
        if args.batch not in [9, 10, 11]:
            raise Exception('Carmel batch No. must be one of 9, 10 or 11')

        key = 'Carmel ' + str(args.batch)  # TODO: Modify this
        test_data_dir = test_data_dir[key]

    elif dset == 'TCGA_ABCTB->CARMEL' or dset == 'CARMEL->CARMEL 9-11':
        key = 'CARMEL' if args.batch == 1 else 'CARMEL 9-11'
        
        if key == 'CARMEL':
            file_name_extension = '_Carmel1-8'
        dset = key
            
        key = key if key in test_data_dir.keys() else 'CARMEL' + str(args.batch)

        test_data_dir = test_data_dir[key] # For specific TCGA_ABCTB->CARMEL fold None case

elif args.haemek_test_set:
    key = 'HAEMEK'
elif args.TA_test_set:
    key = 'TA_fold6'
    dset = 'TCGA_ABCTB'
elif args.PORTUGAL_test_set:
    key = 'PORTUGAL'
    dset = 'PORTUGAL'
elif args.SHEBA_test_set:
    key = 'SHEBA_fold6_onco_features'
    dset = 'SHEBA'
elif args.hic_test_set:
    key = 'HIC'
else:
    key = ''
    
# Fix target:
if target == 'ER_for_is_Tumor_Features':
    target = 'ER_Features'
    
# Get data:
if dataset == 'Combined Features':
    inf_dset = datasets.Combined_Features_for_MIL_Training_dataset(
        dataset_list=['CAT', 'CARMEL'],
        is_all_tiles=True,
        target=target,
        is_train=False,
        test_fold=test_fold,
        is_per_patient=args.is_per_patient)

elif dataset == 'Combined Features - Multi Resolution':
    inf_dset = datasets.Combined_Features_for_MIL_Training_dataset(
        dataset_list=['CARMEL', 'CARMEL_40'],
        is_all_tiles=True,
        target=target,
        is_train=False,
        test_fold=test_fold,
        is_per_patient=args.is_per_patient)

else:
    print(f"test_data_dir is {test_data_dir}")
    if 'OR' in target:
        inf_dset = datasets.Features_MILdataset_combined(dataset=dset,
                                            data_location=test_data_dir,
                                            is_per_patient=args.is_per_patient,
                                            target=target,
                                            is_all_tiles=True,
                                            is_train=False,
                                            test_fold=test_fold,
                                            carmel_only=args.carmel_only)
        is_or = True #flag for extract_tile_scores_for_slide function
    else:
        inf_dset = datasets.Features_MILdataset(dataset=dset,
                                            data_location=test_data_dir,
                                            target=target,
                                            is_per_patient=args.is_per_patient,
                                            is_all_tiles=True,
                                            is_train=False,
                                            carmel_only=args.carmel_only,
                                            test_fold=test_fold)
        is_or = False

inf_loader = DataLoader(inf_dset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=0,
                        pin_memory=True)

#if not (args.carmel_test_set or args.haemek_test_set):
if dset not in ['CARMEL 9-11', 'HAEMEK', 'HIC'] and not (args.TA_test_set or args.SHEBA_test_set or args.PORTUGAL_test_set):
    compute_performance = True
    fig1, ax1 = plt.subplots()
    ax1.set_prop_cycle(custom_cycler)
    legend_labels = []
else:
    compute_performance = False

if args.save_tile_scores and len(args.from_epoch) > 1:
    raise Exception('When saving tile scores, there should be only one model')

# When saving the data we save data for all features (including carmel) so there is no need to save it again when
# working only on carmel slides
if args.save_tile_scores and not args.carmel_only:
    all_slides_weights_before_sftmx_list = []
    all_slides_weights_after_sftmx_list = []
    all_slides_tile_scores_list = []
    all_slides_scores_list = []

    all_slides_weights_before_sftmx_list.append({})
    all_slides_weights_after_sftmx_list.append({})
    all_slides_tile_scores_list.append({})
    all_slides_scores_list.append({})

    if dataset in [
            'Combined Features', 'Combined Features - Multi Resolution'
    ]:
        all_slides_tile_scores_list_ran = {
            inf_dset.dataset_list[0]: [{}],
            inf_dset.dataset_list[1]: [{}]
        }
        all_slides_scores_list_ran = {
            inf_dset.dataset_list[0]: [{}],
            inf_dset.dataset_list[1]: [{}]
        }
    else:
        all_slides_tile_scores_list_ran = []
        all_slides_scores_list_ran = []
        all_slides_tile_scores_list_ran.append({})
        all_slides_scores_list_ran.append({})

# Load model
model = eval(model_name)
if free_bias:
    model.create_free_bias()
if CAT_only:
    model.Model_1_only = True

if hasattr(args, 'target') and '+is_Tumor' in args.target:
    model.set_isTumor_train_mode(isTumor_train_mode=is_tumor_train_mode)

total_loss, total_tiles_infered = 0, 0
for model_num, model_epoch in enumerate(args.from_epoch):
    model_data_loaded = torch.load(os.path.join(
        output_dir, 'Model_CheckPoints',
        'model_data_Epoch_' + str(model_epoch) + '.pt'),
                                   map_location='cpu')

    model.load_state_dict(model_data_loaded['model_state_dict'])

    if dataset in [
            'Combined Features', 'Combined Features - Multi Resolution'
    ] and model_data_loaded['bias_Model_1'] != None:
        model.bias_Model_1 = model_data_loaded['bias_Model_1']
        model.bias_Model_2 = model_data_loaded['bias_Model_2']
        if 'class relation' in model_data_loaded.keys(
        ) and model_data_loaded['class relation'] != None:
            model.relation = model_data_loaded['class relation']

    scores_reg = [] if dataset not in [
        'Combined Features', 'Combined Features - Multi Resolution'
    ] else {
        inf_dset.dataset_list[0]: [],
        inf_dset.dataset_list[1]: []
    }
    all_scores_mil, all_labels_mil, all_targets = [], [], []
    total, correct_pos, correct_neg = 0, 0, 0
    total_pos, total_neg = 0, 0
    true_targets, scores_mil = np.zeros(0), np.zeros(0)
    correct_labeling = 0

    model.to(DEVICE)
    model.infer = True
    model.eval()

    with torch.no_grad():
        for batch_idx, minibatch in enumerate(tqdm(inf_loader)):
            if dataset in [
                    'Combined Features', 'Combined Features - Multi Resolution'
            ]:
                total_tiles_infered += minibatch[list(
                    minibatch.keys())[0]]['tile scores'].size(
                        1)  #  count the number of tiles in each minibatch
                target = minibatch[inf_dset.dataset_list[0]]['targets']
                data_Model_1 = minibatch[inf_dset.dataset_list[0]]['features']
                data_Model_2 = minibatch[inf_dset.dataset_list[1]]['features']

                data_Model_1, data_Model_2, target = data_Model_1.to(
                    DEVICE), data_Model_2.to(DEVICE), target.to(DEVICE)
                data = {
                    inf_dset.dataset_list[0]: data_Model_1,
                    inf_dset.dataset_list[1]: data_Model_2
                }
            else:
                total_tiles_infered += minibatch['tile scores'].size(1)
                target = minibatch['targets']
                data = minibatch['features']
                if hasattr(args, 'target') and '+is_Tumor' in args.target:
                    # conctenating both data vectors:
                    data = torch.cat((data, minibatch['tumor_features']),
                                     axis=2)

                data, target = data.to(DEVICE), target.to(DEVICE)

            if model_num == 0:
                if dataset in [
                        'Combined Features',
                        'Combined Features - Multi Resolution'
                ]:
                    scores_reg[inf_dset.dataset_list[0]].append(
                        minibatch[inf_dset.dataset_list[0]]
                        ['tile scores'].mean().cpu().item())
                    scores_reg[inf_dset.dataset_list[1]].append(
                        minibatch[inf_dset.dataset_list[1]]
                        ['tile scores'].mean().cpu().item())
                else:
                    scores_reg.append(minibatch['scores'].mean().cpu().item())

            outputs, weights_after_sftmx, weights_before_sftmx = model(x=None,
                                                                       H=data)

            #if not (args.carmel_test_set or args.haemek_test_set):  # This is fםr use in CARMEL Batch 9-11 where the targets are unknown and where given -1
            if compute_performance:
                minibatch_loss = criterion(outputs, target)
                total_loss += minibatch_loss

            if dataset in [
                    'Combined Features', 'Combined Features - Multi Resolution'
            ]:
                if type(
                        weights_after_sftmx
                ) == list:  # This will work on the model Combined_MIL_Feature_Attention_MultiBag_DEBUG
                    if len(weights_after_sftmx) == 2:
                        weights_after_sftmx = {
                            inf_dset.dataset_list[0]:
                            weights_after_sftmx[0].cpu().detach().numpy(),
                            inf_dset.dataset_list[1]:
                            weights_after_sftmx[1].cpu().detach().numpy()
                        }
                    elif len(weights_after_sftmx) == 1:
                        weights_after_sftmx = {
                            'CAT':
                            weights_after_sftmx[0].cpu().detach().numpy(),
                            'CARMEL': None
                        }

                else:
                    for key in list(weights_after_sftmx.keys()):
                        weights_after_sftmx[key] = weights_after_sftmx[
                            key].cpu().detach().numpy()
                        weights_before_sftmx[key] = weights_before_sftmx[
                            key].cpu().detach().numpy()
            else:
                weights_after_sftmx = weights_after_sftmx.cpu().detach().numpy(
                )
                weights_before_sftmx = weights_before_sftmx.cpu().detach(
                ).numpy()

            outputs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            if args.save_tile_scores and not args.carmel_only:
                if dataset in [
                        'Combined Features',
                        'Combined Features - Multi Resolution'
                ]:
                    for key in list(minibatch.keys()):
                        slide_name = minibatch[key]['slide name']
                        tile_scores_ran = minibatch[key]['tile scores'].cpu(
                        ).detach().numpy()[0]

                        all_slides_scores_list_ran[key][model_num][
                            slide_name[0]] = minibatch[key][
                                'slide scores'].cpu().detach().numpy()

                        if len(tile_scores_ran) != 500:
                            new_tile_scores_ran = np.zeros(500, )
                            new_tile_scores_ran[:len(tile_scores_ran
                                                     ), ] = tile_scores_ran
                            tile_scores_ran = new_tile_scores_ran

                        all_slides_tile_scores_list_ran[key][model_num][
                            slide_name[0]] = tile_scores_ran

                        features_to_save = torch.transpose(
                            data[key].squeeze(0), 1, 0)
                        slide_tile_scores_list = utils_MIL.extract_tile_scores_for_slide(
                            features_to_save, [model], is_or)

                        if len(slide_tile_scores_list[0]) != 500:
                            new_slide_tile_scores_list = np.zeros(500, )
                            new_slide_tile_scores_list[:len(
                                slide_tile_scores_list[0]
                            ), ] = slide_tile_scores_list[0]
                            slide_tile_scores_list[
                                0] = new_slide_tile_scores_list

                        if weights_after_sftmx.shape[1] != 500:
                            new_weights = np.zeros((1, 500))
                            new_weights[:, :weights_after_sftmx.
                                        shape[1]] = weights_after_sftmx
                            weights_after_sftmx = new_weights

                        if weights_before_sftmx.shape[1] != 500:
                            new_weights = np.zeros((1, 500))
                            new_weights[:, :weights_before_sftmx.
                                        shape[1]] = weights_before_sftmx
                            weights_before_sftmx = new_weights

                        all_slides_tile_scores_list[model_num][
                            slide_name[0]] = slide_tile_scores_list[0]
                        all_slides_weights_before_sftmx_list[model_num][
                            slide_name[0]] = weights_before_sftmx.reshape(
                                weights_before_sftmx.shape[1], )
                        all_slides_weights_after_sftmx_list[model_num][
                            slide_name[0]] = weights_after_sftmx.reshape(
                                weights_after_sftmx.shape[1], )
                        # all_slides_weights_list[model_num][slide_name[0]] = weights_before_sftmx.reshape(weights_before_sftmx.shape[1], )
                        all_slides_scores_list[model_num][
                            slide_name[0]] = outputs[:,
                                                     1].cpu().detach().numpy()

                else:
                    slide_name = minibatch['slide name']
                    tile_scores_ran = minibatch['tile scores'].cpu().detach(
                    ).numpy()[0]

                    all_slides_scores_list_ran[model_num][slide_name[
                        0]] = minibatch['scores'].cpu().detach().numpy()

                    if len(tile_scores_ran) != 500:
                        new_tile_scores_ran = np.zeros(500, )
                        new_tile_scores_ran[:len(tile_scores_ran
                                                 ), ] = tile_scores_ran
                        tile_scores_ran = new_tile_scores_ran

                    all_slides_tile_scores_list_ran[model_num][
                        slide_name[0]] = tile_scores_ran

                    features_to_save = torch.transpose(data.squeeze(0), 1, 0)
                    slide_tile_scores_list = utils_MIL.extract_tile_scores_for_slide(
                        features_to_save, [model], is_or)

                    if len(slide_tile_scores_list[0]) != 500:
                        new_slide_tile_scores_list = np.zeros(500, )
                        new_slide_tile_scores_list[:len(
                            slide_tile_scores_list[0]
                        ), ] = slide_tile_scores_list[0]
                        slide_tile_scores_list[0] = new_slide_tile_scores_list

                    if weights_after_sftmx.shape[1] != 500:
                        new_weights = np.zeros((1, 500))
                        new_weights[:, :weights_after_sftmx.
                                    shape[1]] = weights_after_sftmx
                        weights_after_sftmx = new_weights

                    if weights_before_sftmx.shape[1] != 500:
                        new_weights = np.zeros((1, 500))
                        new_weights[:, :weights_before_sftmx.
                                    shape[1]] = weights_before_sftmx
                        weights_before_sftmx = new_weights

                    all_slides_tile_scores_list[model_num][
                        slide_name[0]] = slide_tile_scores_list[0]
                    all_slides_weights_before_sftmx_list[model_num][
                        slide_name[0]] = weights_before_sftmx.reshape(
                            weights_before_sftmx.shape[1], )
                    all_slides_weights_after_sftmx_list[model_num][
                        slide_name[0]] = weights_after_sftmx.reshape(
                            weights_after_sftmx.shape[1], )
                    #all_slides_weights_list[model_num][slide_name[0]] = weights_before_sftmx.reshape(weights_before_sftmx.shape[1], )
                    all_slides_scores_list[model_num][
                        slide_name[0]] = outputs[:, 1].cpu().detach().numpy()

            scores_mil = np.concatenate(
                (scores_mil, outputs[:, 1].cpu().detach().numpy()))
            #if not (args.carmel_test_set or args.haemek_test_set):  # We dont care about targets when doing true tests (on carmel 9-11)
            if compute_performance:
                true_targets = np.concatenate(
                    (true_targets, target.cpu().detach().numpy()))

                total += target.size(0)
                total_pos += target.eq(1).sum().item()
                total_neg += target.eq(0).sum().item()
                correct_labeling += predicted.eq(target).sum().item()

                correct_pos += predicted[target.eq(1)].eq(1).sum().item()
                correct_neg += predicted[target.eq(0)].eq(0).sum().item()

                all_targets.append(target.cpu().detach().numpy().item())

            all_labels_mil.append(predicted.cpu().detach().numpy().item())
            all_scores_mil.append(outputs[:, 1].cpu().detach().numpy().item())

    if args.save_tile_scores and not args.carmel_only:
        all_slides_score_dict = {
            'MIL': all_slides_scores_list,
            'REG': all_slides_scores_list_ran
        }
        all_tile_scores_dict = {
            'MIL': all_slides_tile_scores_list,
            'REG': all_slides_tile_scores_list_ran
        }
        utils_MIL.save_all_slides_and_models_data(
            all_tile_scores_dict,
            all_slides_score_dict,
            all_slides_weights_before_sftmx_list,
            all_slides_weights_after_sftmx_list, [model],
            output_dir,
            args.from_epoch,
            '',
            true_test_path=key,
            sub_dir='')

    #if not (args.carmel_test_set or args.haemek_test_set):  # We can skip this part when working with true test
    if compute_performance:
        if model_num == 0:
            if dataset in [
                    'Combined Features', 'Combined Features - Multi Resolution'
            ]:
                fpr_reg, tpr_reg, roc_auc_reg = {}, {}, {}
                fpr_reg[inf_dset.dataset_list[0]], tpr_reg[
                    inf_dset.dataset_list[0]], _ = roc_curve(
                        true_targets,
                        np.array(scores_reg[inf_dset.dataset_list[0]]))
                fpr_reg[inf_dset.dataset_list[1]], tpr_reg[
                    inf_dset.dataset_list[1]], _ = roc_curve(
                        true_targets,
                        np.array(scores_reg[inf_dset.dataset_list[1]]))
                roc_auc_reg[inf_dset.dataset_list[0]] = auc(
                    fpr_reg[inf_dset.dataset_list[0]],
                    tpr_reg[inf_dset.dataset_list[0]])
                roc_auc_reg[inf_dset.dataset_list[1]] = auc(
                    fpr_reg[inf_dset.dataset_list[1]],
                    tpr_reg[inf_dset.dataset_list[1]])
                plt.plot(fpr_reg[inf_dset.dataset_list[0]],
                         tpr_reg[inf_dset.dataset_list[0]])
                plt.plot(fpr_reg[inf_dset.dataset_list[1]],
                         tpr_reg[inf_dset.dataset_list[1]])
            else:
                fpr_reg, tpr_reg, _ = roc_curve(true_targets,
                                                np.array(scores_reg))
                roc_auc_reg = auc(fpr_reg, tpr_reg)
                plt.plot(fpr_reg, tpr_reg)

            postfix = 'Patient' if args.is_per_patient else 'Slide'
            if dataset in [
                    'Combined Features', 'Combined Features - Multi Resolution'
            ]:
                for key in list(minibatch.keys()):
                    label_reg = 'REG [' + key + '] Per ' + postfix + ' AUC='
                    legend_labels.append(label_reg +
                                         str(round(roc_auc_reg[key] *
                                                   100, 2)) + '%)')
            else:
                label_reg = 'REG Per ' + postfix + ' AUC='
                legend_labels.append(label_reg +
                                     str(round(roc_auc_reg * 100, 2)) + '%)')

            label_MIL = 'Model' + str(
                model_epoch) + ': MIL Per ' + postfix + ' AUC='

        fpr_mil, tpr_mil, _ = roc_curve(true_targets, scores_mil)
        roc_auc_mil = auc(fpr_mil, tpr_mil)
        plt.plot(fpr_mil, tpr_mil)
        legend_labels.append(label_MIL + str(round(roc_auc_mil * 100, 2)) +
                             '%)')

#if not (args.carmel_test_set or args.haemek_test_set):
if compute_performance:
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(legend_labels)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(b=True)
    title = 'Per {}, Model {}.'.format(
        'Patient' if args.is_per_patient else 'Slide', args.experiment)
    if args.is_per_patient:
        title += ' Removed {}/{} Patients.'.format(
            len(inf_dset.bad_patient_list), len(inf_dset.patient_set))

    plt.title(title)

    if args.is_per_patient:
        graph_name = 'feature_mil_inference_per_patient_CARMEL_ONLY.jpg' if args.carmel_only else 'feature_mil_inference_per_patient.jpg'
    else:
        graph_name = 'feature_mil_inference_per_slide_CARMEL_ONLY.jpg' if args.carmel_only else 'feature_mil_inference_per_slide.jpg'

    if 'file_name_extension' in locals():
        graph_name = '.'.join(graph_name.split(
            '.')[:-1]) + file_name_extension + graph_name.split('.')[-1]

    plt.savefig(os.path.join(output_dir, graph_name))

print('Done')