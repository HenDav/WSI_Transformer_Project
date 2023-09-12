"""
This file save tile scores on excel files.
It differs from the file 'inference_Multi_REG.py' by that it saves only the tile scores.
"""

import utils
from torch.utils.data import DataLoader
import torch
import datasets
import numpy as np
import os
import sys, platform
import argparse
from tqdm import tqdm
from collections import OrderedDict
from Nets import resnet_v2
import pandas as pd

parser = argparse.ArgumentParser(description='WSI_REG Slide inference')
parser.add_argument('-ex', '--experiment', nargs='+', type=int, default=[10607], help='Use models from this experiment')
parser.add_argument('-fe', '--from_epoch', nargs='+', type=int, default=[960], help='Use this epoch models for inference')
parser.add_argument('-nt', '--num_tiles', type=int, default=60, help='Number of tiles to use')
parser.add_argument('-ds', '--dataset', type=str, default='ABCTB', help='DataSet to use')
parser.add_argument('-f', '--folds', type=list, nargs="+", default=1, help=' folds to infer')
parser.add_argument('--mag', type=int, default=10, help='desired magnification of patches')
parser.add_argument('-mp', '--model_path', type=str, default='', help='fixed path of rons model')
parser.add_argument('-d', dest='dx', action='store_true', help='Use ONLY DX cut slides')
parser.add_argument('--patch_dir', type=str, default='', help='patch locations directory, for use with predecided patches')
parser.add_argument('-sd', '--subdir', type=str, default='', help='output sub-dir')
parser.add_argument('--resume', type=int, default=0, help='resume a failed feature extraction')
args = parser.parse_args()

if type(args.folds) != int:
    args.folds = list(map(int, args.folds[0])) #RanS 14.6.21

# If args.experiment contains 1 number than all epochs are from the same experiments, BUT if it is bigger than 1 than all
# the length of args.experiment should be equal to args.from_epoch
if len(args.experiment) > 1:
    if len(args.experiment) != len(args.from_epoch):
        raise Exception("number of from_epoch(-fe) should be equal to number of experiment(-ex)")
    else:
        different_experiments = True
        Output_Dirs = []
else:
    different_experiments = False

DEVICE = utils.device_gpu_cpu()
data_path = ''

print('Loading pre-saved models:')
models = []
dx = False

#decide which epochs to save features from - if model_path is used, take it.

for counter in range(len(args.from_epoch)):
    epoch = args.from_epoch[counter]
    experiment = args.experiment[counter] if different_experiments else args.experiment[0]

    print('  Exp. {} and Epoch {}'.format(experiment, epoch))
    # Basic meta data will be taken from the first model (ONLY if all inferences are done from the same experiment)
    if counter == 0:
        run_data_output = utils.run_data(experiment=experiment)
        output_dir, TILE_SIZE, dx, args.target, model_name, args.mag =\
            run_data_output['Location'], run_data_output['Tile Size'], run_data_output['DX'], run_data_output['Receptor'],\
            run_data_output['Model Name'], run_data_output['Desired Slide Magnification']
        if different_experiments:
            Output_Dirs.append(output_dir)
        fix_data_path = True
    elif counter > 0 and different_experiments:
        run_data_output = utils.run_data(experiment=experiment)
        output_dir, dx, target, model_name, args.mag =\
            run_data_output['Location'], run_data_output['DX'], run_data_output['Receptor'],\
            run_data_output['Model Name'], run_data_output['Desired Slide Magnification']
        Output_Dirs.append(output_dir)
        fix_data_path = True

    # fix target:
    if args.target in ['Features_Survival_Time_Cox', 'Features_Survival_Time_L2', 'Survival_Time_Cox']:
        args.target = 'survival'
        survival_kind = 'Time'
        target_for_file_name = args.target + '_' + survival_kind
    else:
        survival_kind = 'Binary'

    if fix_data_path:
        # we need to make some root modifications according to the computer we're running at.
        if sys.platform == 'linux':
            data_path = ''

        elif sys.platform == 'win32':
            output_dir = output_dir.replace(r'/', '\\')
            data_path = os.getcwd()

        elif sys.platform == 'darwin':
            output_dir = '/'.join(output_dir.split('/')[4:])
            data_path = os.getcwd()

        fix_data_path = False

        # Verifying that the target receptor is not changed:
        if counter > 1 and args.target != target:
            raise Exception("Target Receptor is changed between models - DataSet cannot support this action")

    if len(args.target.split('+')) > 1:
        multi_target = True
        target0, target1 = args.target.split('+')
        model_name = model_name[:-2] + '(num_classes=4)' #manually add num_classes since the arguments are not saved in run_data
    else:
        multi_target = False

    # loading basic model type
    model = eval(model_name)
    # loading model parameters from the specific epoch
    model_data_loaded = torch.load(os.path.join(data_path, output_dir, 'Model_CheckPoints',
                                                'model_data_Epoch_' + str(epoch) + '.pt'), map_location='cpu')
    # Making sure that the size of the linear layer of the loaded model, fits the basic model.
    model.linear = torch.nn.Linear(in_features=model_data_loaded['model_state_dict']['linear.weight'].size(1),
                                   out_features=model_data_loaded['model_state_dict']['linear.weight'].size(0))

    model.load_state_dict(model_data_loaded['model_state_dict'])
    model.eval()
    models.append(model)

# override run_data dx if args.dx is true
if args.dx:
    dx = args.dx

TILE_SIZE = 128
tiles_per_iter = 20
if sys.platform == 'linux':
    TILE_SIZE = 256
    tiles_per_iter = 150
    if platform.node() in ['gipdeep4', 'gipdeep5', 'gipdeep6']:
        tiles_per_iter = 100
elif sys.platform == 'win32':
    TILE_SIZE = 256

#RanS 16.3.21, support ron's model as well
if args.model_path != '':
    if os.path.exists(args.model_path):
        args.from_epoch.append('rons_model')
        model = resnet_v2.PreActResNet50()
        model_data_loaded = torch.load(os.path.join(args.model_path), map_location='cpu')

        try:
            model.load_state_dict(model_data_loaded['net'])
        except:
            state_dict = model_data_loaded['net']
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove 'module.' of dataparallel
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
    else:
        #RanS 27.10.21, use pretrained model
        args.from_epoch.append(args.model_path.split('.')[-1])
        model = eval(args.model_path)
        model.fc = torch.nn.Identity()
        tiles_per_iter = 100
    model.eval()
    models.append(model)

slide_num = args.resume

inf_dset = datasets.Infer_Dataset_Survival(DataSet=args.dataset,
                                           tile_size=TILE_SIZE,
                                           tiles_per_iter=tiles_per_iter,
                                           target_kind=args.target,
                                           folds=args.folds,
                                           num_tiles=args.num_tiles,
                                           desired_slide_magnification=args.mag,
                                           dx=dx,
                                           resume_slide=slide_num,
                                           patch_dir=args.patch_dir
                                           )

inf_loader = DataLoader(inf_dset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

new_slide = True

NUM_MODELS = len(models)
NUM_SLIDES = len(inf_dset.image_file_names)
print('NUM_SLIDES: ', str(NUM_SLIDES))

tile_scores = np.zeros(args.num_tiles) * np.nan
all_slide_names, all_targets, all_targets_time, all_targets_binary, all_censored = [], [], [], [], []
tile_dict = {}
target_dict = {}

if not os.path.isdir(os.path.join(data_path, output_dir, 'Inference')):
    os.mkdir(os.path.join(data_path, output_dir, 'Inference'))

if not os.path.isdir(os.path.join(data_path, output_dir, 'Inference', args.subdir)):
    os.mkdir(os.path.join(data_path, output_dir, 'Inference', args.subdir))

print('slide_num0 = ', str(slide_num)) #temp
with torch.no_grad():
    for batch_idx, MiniBatch_Dict in enumerate(tqdm(inf_loader)):

        # Unpacking the data:
        data = MiniBatch_Dict['Data']
        target = MiniBatch_Dict['Label']
        last_batch = MiniBatch_Dict['Is Last Batch']
        slide_file = MiniBatch_Dict['Slide Filename']
        slide_dataset = MiniBatch_Dict['Slide DataSet']
        patch_locs = MiniBatch_Dict['Patch Loc']
        patient = MiniBatch_Dict['Patient barcode'][0]

        if args.target == 'survival':
            target_time = MiniBatch_Dict['Time Target']
            target_binary = MiniBatch_Dict['Binary Target']
            censored = MiniBatch_Dict['Censored']

        if new_slide:
            n_tiles = inf_loader.dataset.num_tiles[slide_num - args.resume]
            scores_0 = [np.zeros(n_tiles) for ii in range(NUM_MODELS)]
            scores_1 = [np.zeros(n_tiles) for ii in range(NUM_MODELS)]

            patch_locs_1_slide = np.zeros((n_tiles, 2))
            slide_batch_num = 0
            new_slide = False

        data = data.squeeze(0)
        data, target = data.to(DEVICE), target.to(DEVICE)

        patch_locs_1_slide[slide_batch_num * tiles_per_iter: slide_batch_num * tiles_per_iter + len(data),:] = np.array(patch_locs)

        for index, model in enumerate(models):
            model.to(DEVICE)

            if model._get_name() == 'PreActResNet_Ron':
                scores, _ = model(data)


            if survival_kind != 'Time':
                scores = torch.nn.functional.softmax(scores, dim=1)

            if survival_kind == 'Time':  # TODO: This can be changed to else and connected to the if statement above
                # The value inside scores_1 is later saved as the score for the tile, so in the case of only 1 score per
                # tile, we'll save it's value.
                tile_scores[slide_batch_num * tiles_per_iter: slide_batch_num * tiles_per_iter + len(data)] = scores[:, 0].cpu().detach().numpy()

            else:  # In case this is not a time model (L2 or Cox) we have more than 1 score per tile:
                raise Exception('Need to implement')

        slide_batch_num += 1

        if last_batch:
            new_slide = True
            tile_dict[slide_file[0]] = list(tile_scores)
            all_targets.append(target.cpu().numpy()[0][0])

            if args.target == 'survival':
                target_dict[slide_file[0]] = {'Patient': patient,
                                              'Time': target_time.item(),
                                              'Binary': target_binary[0].detach().cpu().numpy(),
                                              'Censored': censored.item()}

            slide_num += 1

tile_scores_DF = pd.DataFrame(tile_dict).transpose()
targets_DF = pd.DataFrame(target_dict).transpose()

all_data_DF = pd.concat([targets_DF, tile_scores_DF], axis=1)
file_name = os.path.join(data_path, output_dir, 'Inference', args.subdir, 'Model_' + str(args.experiment[0]) + '_Epoch_' + str(args.from_epoch[0])
                                 + '-Folds_' + str(args.folds) + '_' + target_for_file_name + '-Tiles_' + str(args.num_tiles) + '.xlsx')
all_data_DF.to_excel(file_name)

print('Done !')
