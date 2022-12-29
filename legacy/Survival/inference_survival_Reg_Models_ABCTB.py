import utils
from torch.utils.data import DataLoader
import torch
import datasets
import numpy as np
from sklearn.metrics import roc_curve
import os
import sys, platform
import argparse
from tqdm import tqdm
import pickle
from collections import OrderedDict
from Nets import resnet_v2
from pathlib import Path

parser = argparse.ArgumentParser(description='WSI_REG Slide inference')
parser.add_argument('-ex', '--experiment', nargs='+', type=int, default=[10735], help='Use models from this experiment')
parser.add_argument('-fe', '--from_epoch', nargs='+', type=int, default=[1000], help='Use this epoch models for inference')
parser.add_argument('-nt', '--num_tiles', type=int, default=500, help='Number of tiles to use')
parser.add_argument('-ds', '--dataset', type=str, default='ABCTB', help='DataSet to use')
parser.add_argument('-f', '--folds', type=int, nargs="+", default=2, help=' folds to infer')
parser.add_argument('--mag', type=int, default=10, help='desired magnification of patches')
parser.add_argument('-mp', '--model_path', type=str, default='', help='fixed path of rons model')  # r'/home/rschley/Pathnet/results/fold_1_ER_large/checkpoint/ckpt_epoch_1467.pth'
parser.add_argument('-d', dest='dx', action='store_true', help='Use ONLY DX cut slides')  # overrides run_data
parser.add_argument('--patch_dir', type=str, default='', help='patch locations directory, for use with predecided patches')
parser.add_argument('-sd', '--subdir', type=str, default='', help='output sub-dir')
args = parser.parse_args()

if type(args.folds) != int:
    args.folds = list(map(int, args.folds[0]))

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
    if args.target in ['Features_Survival_Time_Cox', 'Features_Survival_Time_L2', 'Survival_Time_Cox', 'Survival_Time_L2']:
        args.target = 'survival'
        survival_kind = 'Time'
        target_for_file_name = args.target + '_' + survival_kind

    elif args.target == 'Survival_Binary':
        args.target = 'survival'
        survival_kind = 'Binary'
        target_for_file_name = args.target + '_' + survival_kind

    elif args.target == 'survival' and survival_kind in ['Time', 'Binary']:
        pass  # do nothing

    elif args.target == 'Survival_Combined_Loss':
        args.target = 'survival'
        survival_kind = 'Survival_Combined_Loss'

    else:
        raise Exception('this target is not implemented yet')

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

# get number of classes based on the first model
N_classes = models[0].linear.out_features #for resnets and such

# override run_data dx if args.dx is true
if args.dx:
    dx = args.dx

if sys.platform == 'linux':
    TILE_SIZE = 256
    tiles_per_iter = 150
    if platform.node() in ['gipdeep4', 'gipdeep5', 'gipdeep6']:
        tiles_per_iter = 100
elif sys.platform == 'win32':
    TILE_SIZE = 256
elif sys.platform == 'darwin':
    TILE_SIZE = 128
    tiles_per_iter = 20

# support ron's model as well
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
        # use pretrained model
        args.from_epoch.append(args.model_path.split('.')[-1])
        model = eval(args.model_path)
        model.fc = torch.nn.Identity()
        tiles_per_iter = 100
    model.eval()
    models.append(model)

slide_num = 0

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
NUM_SLIDES = len(inf_dset.image_file_names)  # valid_slide_indices always counts non-dx slides
NUM_SLIDES_SAVE = 50
print('NUM_SLIDES: ', str(NUM_SLIDES))

# Initializing empty variables to further contain results
all_binary_targets, all_time_targets, all_censored = [], [], []
patch_locs_all = np.zeros((NUM_SLIDES, args.num_tiles, 2))
patch_locs_all[:] = np.nan
all_slide_names = np.zeros(NUM_SLIDES, dtype=object)
all_slide_datasets = np.zeros(NUM_SLIDES, dtype=object)


if survival_kind == 'Survival_Combined_Loss':
    patch_scores = np.empty((NUM_SLIDES, NUM_MODELS, args.num_tiles, N_classes))
    patch_scores_before_softmax = np.empty((NUM_SLIDES, NUM_MODELS, args.num_tiles, N_classes))

    all_scores = np.zeros((NUM_SLIDES, NUM_MODELS, N_classes))
    all_scores_before_softmax = np.zeros((NUM_SLIDES, NUM_MODELS, N_classes))

else:
    all_scores = np.zeros((NUM_SLIDES, NUM_MODELS))  # fixme: this might be changed for combined loss
    patch_scores = np.empty((NUM_SLIDES, NUM_MODELS, args.num_tiles))  # fixme: this might be changed for combined loss

    all_scores_before_softmax = np.zeros((NUM_SLIDES, NUM_MODELS))  # fixme: this might be changed for combined loss
    patch_scores_before_softmax = np.empty((NUM_SLIDES, NUM_MODELS, args.num_tiles))  # fixme: this might be changed for combined loss


patch_scores[:] = np.nan
patch_scores_before_softmax[:] = np.nan

if not os.path.isdir(os.path.join(data_path, output_dir, 'Inference', args.subdir)):
    Path(os.path.join(data_path, output_dir, 'Inference', args.subdir)).mkdir(parents=True)

print('slide_num0 = ', str(slide_num)) #temp
with torch.no_grad():
    for batch_idx, MiniBatch_Dict in enumerate(tqdm(inf_loader)):

        # Unpacking the data:
        data = MiniBatch_Dict['Data']
        time_list = MiniBatch_Dict['Time List']
        last_batch = MiniBatch_Dict['Is Last Batch']
        slide_file = MiniBatch_Dict['Slide Filename']
        slide_dataset = MiniBatch_Dict['Slide DataSet']
        patch_locs = MiniBatch_Dict['Patch Loc']
        target_time = MiniBatch_Dict['Time Target']
        target_binary = MiniBatch_Dict['Binary Target']
        censored = MiniBatch_Dict['Censored']

        if new_slide:
            n_tiles = inf_loader.dataset.num_tiles[slide_num]
            current_slide_tile_scores = [np.zeros((n_tiles, N_classes)) for ii in range(NUM_MODELS)]
            current_slide_tile_scores_before_softmax = [np.zeros((n_tiles, N_classes)) for ii in range(NUM_MODELS)]
            patch_locs_1_slide = np.zeros((n_tiles, 2))

            slide_batch_num = 0
            new_slide = False

        data = data.squeeze(0)
        data = data.to(DEVICE)

        target = target_binary.to(DEVICE)

        patch_locs_1_slide[slide_batch_num * tiles_per_iter: slide_batch_num * tiles_per_iter + len(data),:] = np.array(patch_locs)

        for model_ind, model in enumerate(models):
            model.to(DEVICE)

            if model._get_name() == 'PreActResNet_Ron':
                scores_before_sftmax, features = model(data)

            else:
                raise IOError('Net not supported yet for feature and score exctration, implement!')

            if survival_kind == 'Binary':
                scores = torch.nn.functional.softmax(scores_before_sftmax, dim=1)

            elif survival_kind == 'Survival_Combined_Loss':
                scores = torch.zeros_like(scores_before_sftmax)
                scores[:, :2] = scores_before_sftmax[:, :2]
                scores[:, 2:] = torch.nn.functional.softmax(scores_before_sftmax[:, 2:], dim=1)

            else:
                scores = scores_before_sftmax

            current_slide_tile_scores[model_ind][slide_batch_num * tiles_per_iter: slide_batch_num * tiles_per_iter + len(data), :] = scores.cpu().detach().numpy()
            current_slide_tile_scores_before_softmax[model_ind][slide_batch_num * tiles_per_iter: slide_batch_num * tiles_per_iter + len(data), :] = scores_before_sftmax.cpu().detach().numpy()

        slide_batch_num += 1

        if last_batch:
            new_slide = True

            all_binary_targets.append(target_binary.cpu().numpy()[0][0])
            all_time_targets.append(target_time.cpu().numpy()[0])
            all_censored.append(censored.cpu().numpy()[0])

            patch_locs_all[slide_num, :len(patch_locs_1_slide), :] = patch_locs_1_slide

            for model_ind in range(NUM_MODELS):
                if survival_kind == 'Survival_Combined_Loss':
                    patch_scores[slide_num, model_ind, :n_tiles, :] = current_slide_tile_scores[model_ind]
                    patch_scores_before_softmax[slide_num, model_ind, :n_tiles, :] = current_slide_tile_scores_before_softmax[model_ind]
                    all_scores[slide_num, model_ind] = current_slide_tile_scores[model_ind].mean(axis=0)
                    all_scores_before_softmax[slide_num, model_ind] = current_slide_tile_scores_before_softmax[model_ind].mean(axis=0)

                else:
                    patch_scores[slide_num, model_ind, :n_tiles] = current_slide_tile_scores[model_ind][:, N_classes - 1]
                    all_scores[slide_num, model_ind] = current_slide_tile_scores[model_ind][:, N_classes - 1].mean()

                    patch_scores_before_softmax[slide_num, model_ind, :n_tiles] = current_slide_tile_scores_before_softmax[model_ind][:, N_classes - 1]
                    all_scores_before_softmax[slide_num, model_ind] = current_slide_tile_scores_before_softmax[model_ind][:, N_classes - 1].mean()

                all_slide_names[slide_num] = slide_file[0]
                all_slide_datasets[slide_num] = slide_dataset[0]

            slide_num += 1

for model_num in range(NUM_MODELS):
    if different_experiments:
        output_dir = Output_Dirs[model_num]

    # Computing AUC:
    # remove targets = -1 from auc calculation
    try:
        binary_targets_arr = np.array(all_binary_targets)
        time_targets_arr = np.array(all_time_targets)
        censored_arr = np.array(all_censored)

        if survival_kind == 'Survival_Combined_Loss':
            scores_arr = None
            fpr, tpr = 0, 0
        else:
            scores_arr = all_scores[:, model_num]
            fpr, tpr, _ = roc_curve(binary_targets_arr[binary_targets_arr >= 0], scores_arr[binary_targets_arr >= 0])

    except ValueError:
        fpr, tpr = 0, 0  # if all labels are unknown

    # Save inference data to file:
    file_name = os.path.join(data_path, output_dir, 'Inference', args.subdir, 'Model_Epoch_' + str(args.from_epoch[model_num])
                             + '-Folds_' + str(args.folds) + '_' + str(args.target) + '-Tiles_' + str(args.num_tiles) + '.data')

    inference_data = [fpr, tpr, all_scores[:, model_num], NUM_SLIDES,
                      np.squeeze(patch_scores[:, model_num, :]), all_slide_names, all_slide_datasets,
                      np.squeeze(patch_locs_all), binary_targets_arr, time_targets_arr, censored_arr,
                      all_scores_before_softmax[:, model_num], np.squeeze(patch_scores_before_softmax[:, model_num, :])]

    with open(file_name, 'wb') as filehandle:
        pickle.dump(inference_data, filehandle)

    experiment = args.experiment[model_num] if different_experiments else args.experiment[0]

print('Done !')
