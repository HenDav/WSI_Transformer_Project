import datasets
import argparse
from torch.utils.data import DataLoader
import torch
import numpy as np
from Survival.Cox_Loss import Cox_loss
from sksurv.metrics import concordance_index_censored
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
import utils
import os
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from sklearn.metrics import roc_curve, auc, roc_auc_score
import pandas as pd
import sys

parser = argparse.ArgumentParser(description='WSI_REG Training of PathNet Project')
parser.add_argument('-tf', '--test_fold', default=1, type=int, help='fold to be as TEST FOLD')
parser.add_argument('-e', '--epochs', default=1001, type=int, help='Epochs to run')
parser.add_argument('-ex', '--experiment', type=int, default=0, help='Continue train of this experiment')
parser.add_argument('-fe', '--from_epoch', type=int, default=0, help='Continue train from epoch')
parser.add_argument('-d', dest='dx', action='store_true', help='Use ONLY DX cut slides')
parser.add_argument('-ds', '--dataset', type=str, default='ABCTB', help='DataSet to use')
parser.add_argument('-time', dest='time', action='store_true', help='save train timing data ?')
parser.add_argument('-tar', '--target', default='Survival_Binary', type=str, help='label: Her2/ER/PR/EGFR/PDL1')
parser.add_argument('--n_patches_test', default=1, type=int, help='# of patches at test time')
parser.add_argument('--n_patches_train', default=10, type=int, help='# of patches at train time')
parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=5e-5, type=float, help='L2 penalty')
parser.add_argument('--transform_type', default='rvf', type=str, help='none / flip / wcfrs (weak color+flip+rotate+scale)')
parser.add_argument('--batch_size', default=18, type=int, help='size of batch')
parser.add_argument('--model', default='PreActResNets.PreActResNet50_Ron()', type=str, help='net to use')
parser.add_argument('--eval_rate', type=int, default=5, help='Evaluate validation set every # epochs')
parser.add_argument('--c_param', default=0.1, type=float, help='color jitter parameter')
parser.add_argument('-im', dest='images', action='store_true', help='save data images?')
parser.add_argument('--mag', type=int, default=10, help='desired magnification of patches')
parser.add_argument('--loan', action='store_true', help='Localized Annotation for strongly supervised training') #RanS 17.6.21
parser.add_argument('--er_eq_pr', action='store_true', help='while training, take only er=pr examples') #RanS 27.6.21
parser.add_argument('--slide_per_block', action='store_true', help='for carmel, take only one slide per block') #RanS 17.8.21
args = parser.parse_args()

EPS = 1e-7
TILE_SIZE = 256

if sys.platform == 'darwin':
    TILE_SIZE = 128

def train(model: nn.Module, dloader_train: DataLoader, dloader_test: DataLoader, DEVICE, optimizer, print_timing: bool=False):
    """
    This function trains the model
    :return:
    """
    if not os.path.isdir(os.path.join('Debugging', 'Model_CheckPoints')):
        Path(os.path.join('Debugging', 'Model_CheckPoints')).mkdir(parents=True)

    writer_folder = os.path.join('Debugging', 'writer')
    all_writer = SummaryWriter(os.path.join(writer_folder, 'all'))

    previous_epoch_loss = 1e5

    for e in range(0, args.epochs):
        if args.target in ['Survival_Time', 'Survival_Binary']:
            all_targets, all_outputs, all_censored, all_cont_targets, all_binary_targets = [], [], [], [], []

        if args.target != 'Survival_Time':
                total, correct_pos, correct_neg = 0, 0, 0
                total_pos_train, total_neg_train = 0, 0
                true_targets_train, scores_train = np.zeros(0), np.zeros(0)
                correct_labeling = 0

        train_loss = 0
        slide_names = []

        model.train()
        model.to(DEVICE)

        print('Starting Epoch No. {}:'.format(e))
        for batch_idx, minibatch in enumerate(tqdm(zip(dloader_train['Censored'], dloader_train['UnCensored']))):  # Omer 7 Nov 2021
            if type(dloader_train) == dict:
                new_minibatch = {}
                for key in minibatch[0].keys():
                    if type(minibatch[0][key]) == torch.Tensor:
                        new_minibatch[key] = torch.cat([minibatch[0][key], minibatch[1][key]], dim=0)
                    elif type(minibatch[0][key]) == list:
                        new_minibatch[key] = minibatch[0][key] + minibatch[1][key]

                minibatch = new_minibatch

            data = minibatch['Data']
            target = minibatch['Target']
            f_names = minibatch['File Names']

            slide_names_batch = [os.path.basename(f_name) for f_name in f_names]
            slide_names.extend(slide_names_batch)

            if args.target in ['Survival_Time', 'Survival_Binary']:
                censored = minibatch['Censored']
                target_binary = minibatch['Target Binary']
                target_cont = minibatch['Survival Time']

            all_targets.extend(target.numpy()[:, 0])
            all_cont_targets.extend(target_cont.numpy())
            all_binary_targets.extend(target_binary.numpy())
            all_censored.extend(censored.numpy())

            data, target = data.to(DEVICE), target.to(DEVICE).squeeze(1)

            optimizer.zero_grad()
            outputs, _ = model(data)

            if args.target == 'Survival_Time':
                loss = criterion(outputs, target, censored)
                outputs = torch.reshape(outputs, [outputs.size(0)])
                all_outputs.extend(outputs.detach().cpu().numpy())

            else:
                loss = criterion(outputs, target)
                outputs = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                all_outputs.extend(- outputs[:, 1].detach().cpu().numpy())

                scores_train = np.concatenate((scores_train, outputs[:, 1].cpu().detach().numpy()))
                true_targets_train = np.concatenate((true_targets_train, target.cpu().detach().numpy()))

                total += target.size(0)
                total_pos_train += target.eq(1).sum().item()
                total_neg_train += target.eq(0).sum().item()
                correct_labeling += predicted.eq(target).sum().item()
                correct_pos += predicted[target.eq(1)].eq(1).sum().item()
                correct_neg += predicted[target.eq(0)].eq(0).sum().item()

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Compute C index
        if args.target in ['Survival_Time', 'Survival_Binary']:
            if len(all_censored) != 0:
                c_index, num_concordant_pairs, num_discordant_pairs, _, _ = concordance_index_censored(np.invert(all_censored), all_cont_targets, all_outputs)
            else:
                print('Zero elements')

        # Compute AUC:
        if args.target == 'Survival_Time':
            # Sorting out all the Censored data
            not_censored_indices = np.where(np.array(all_censored) == False)
            relevant_binary_targets = np.array(all_binary_targets)[not_censored_indices]
            relevant_outputs = np.array(all_outputs)[not_censored_indices]
            # Sorting out all the non valid binary data:
            relevant_binary_targets = np.reshape(relevant_binary_targets, (relevant_binary_targets.shape[0]))
            valid_binary_target_indices = np.where(relevant_binary_targets != -1)
            relevant_binary_targets = relevant_binary_targets[valid_binary_target_indices]
            relevant_outputs = relevant_outputs[valid_binary_target_indices]

            if len(relevant_binary_targets) != 0:
                fpr_train, tpr_train, _ = roc_curve(relevant_binary_targets, - relevant_outputs)
                roc_auc_train = auc(fpr_train, tpr_train)
            else:
                print('Zero elements')

            scores_train, true_targets_train = np.zeros(len(slide_names)), np.zeros(len(slide_names))

        else:
            train_acc = 100 * correct_labeling / total
            balanced_acc_train = 100. * ((correct_pos + EPS) / (total_pos_train + EPS) + (correct_neg + EPS) / (total_neg_train + EPS)) / 2
            roc_auc_train = np.nan
            if not all(true_targets_train == true_targets_train[0]):  #more than one label
                fpr_train, tpr_train, _ = roc_curve(true_targets_train, scores_train)
                roc_auc_train = auc(fpr_train, tpr_train)
            all_writer.add_scalar('Train/Balanced Accuracy', balanced_acc_train, e)
            all_writer.add_scalar('Train/Accuracy', train_acc, e)

        if c_index not in locals():
            c_index = -1

        all_writer.add_scalar('Train/Roc-Auc', c_index, e)
        all_writer.add_scalar('Train/Roc-Auc', roc_auc_train, e)
        all_writer.add_scalar('Train/Loss Per Epoch', train_loss, e)

        print('Finished Epoch: {}, Loss: {:.2f}, Loss Delta: {:.3f}, Train AUC per patch: {:.2f}, C index: {}'
              .format(e,
                      train_loss,
                      previous_epoch_loss - train_loss,
                      roc_auc_train,
                      c_index
                      ))
        previous_epoch_loss = train_loss


        if e % args.eval_rate == 0:
            # Compute slide AUC using slide mean score score:
            patch_df = pd.DataFrame({'slide': slide_names, 'scores': scores_train, 'targets': true_targets_train})
            slide_mean_score_df = patch_df.groupby('slide').mean()

            roc_auc_slide = np.nan
            try:
                if not all(slide_mean_score_df['targets'] == slide_mean_score_df['targets'][0]):  #more than one label
                    roc_auc_slide = roc_auc_score(slide_mean_score_df['targets'], slide_mean_score_df['scores'])
            except KeyError:
                print()

            all_writer.add_scalar('Train/slide AUC', roc_auc_slide, e)
            ############################################################################################################
            # Compute performance over validation set:
            acc_test, bacc_test, roc_auc_test = check_accuracy(model, dloader_test, all_writer, DEVICE, e)

            # Save model to file:
            try:
                model_state_dict = model.module.state_dict()
            except AttributeError:
                model_state_dict = model.state_dict()
            torch.save({'epoch': e,
                        'model_state_dict': model_state_dict,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'tile_size': TILE_SIZE,
                        },
                       os.path.join('Debugging', 'Model_CheckPoints', 'model_data_Epoch_' + str(e) + '.pt'))
            print('saved checkpoint to', 'Debugging')

    all_writer.close()


def check_accuracy(model: nn.Module, data_loader: DataLoader, all_writer, DEVICE, epoch: int):

    if args.target in ['Survival_Time', 'Survival_Binary']:
        all_targets, all_outputs, all_censored, all_cont_targets, all_binary_targets = [], [], [], [], []

    if args.target != 'Survival_Time':
        total_test, true_pos_test, true_neg_test = 0, 0, 0
        total_pos_test, total_neg_test = 0, 0
        true_targets_test, scores_test = np.zeros(0), np.zeros(0)
        correct_labeling_test = 0

    slide_names = []

    model.eval()
    model.to(DEVICE)

    with torch.no_grad():
        for idx, minibatch in enumerate(zip(data_loader['Censored'], data_loader['UnCensored'])):
            if type(data_loader) == dict:
                new_minibatch = {}
                for key in minibatch[0].keys():
                    if type(minibatch[0][key]) == torch.Tensor:
                        new_minibatch[key] = torch.cat([minibatch[0][key], minibatch[1][key]], dim=0)
                    elif type(minibatch[0][key]) == list:
                        new_minibatch[key] = minibatch[0][key] + minibatch[1][key]

                minibatch = new_minibatch

            data = minibatch['Data']
            target = minibatch['Target']
            f_names = minibatch['File Names']

            slide_names_batch = [os.path.basename(f_name) for f_name in f_names]
            slide_names.extend(slide_names_batch)

            if args.target in ['Survival_Time', 'Survival_Binary']:
                censored = minibatch['Censored']
                target_binary = minibatch['Target Binary']
                target_cont = minibatch['Survival Time']

            all_targets.extend(target.numpy()[:, 0])
            all_cont_targets.extend(target_cont.numpy())
            all_binary_targets.extend(target_binary.numpy())
            all_censored.extend(censored.numpy())


            data, targets = data.to(device=DEVICE), target.to(device=DEVICE).squeeze(1)

            outputs, _ = model(data)

            if args.target == 'Survival_Time':
                outputs = torch.reshape(outputs, [outputs.size(0)])
                all_outputs.extend(outputs.detach().cpu().numpy())

            else:
                outputs = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                all_outputs.extend(- outputs[:, 1].detach().cpu().numpy())

                scores_test = np.concatenate((scores_test, outputs[:, 1].cpu().detach().numpy()))
                total_test += targets.size(0)
                total_pos_test += targets.eq(1).sum().item()
                total_neg_test += targets.eq(0).sum().item()
                true_pos_test += predicted[targets.eq(1)].eq(1).sum().item()
                true_neg_test += predicted[targets.eq(0)].eq(0).sum().item()
                correct_labeling_test += predicted.eq(targets).sum().item()
                true_targets_test = np.concatenate((true_targets_test, targets.cpu().detach().numpy()))

        # Compute C index
        if args.target in ['Survival_Time', 'Survival_Binary']:
            if len(all_censored) != 0:
                c_index_test, num_concordant_pairs, num_discordant_pairs, _, _ = concordance_index_censored(np.invert(all_censored), all_cont_targets, all_outputs)
            else:
                print('Zero elements')

        # Compute AUC:
        if args.target == 'Survival_Time':
            # Sorting out all the Censored data
            not_censored_indices = np.where(np.array(all_censored) == False)
            relevant_binary_targets = np.array(all_binary_targets)[not_censored_indices]
            relevant_outputs = np.array(all_outputs)[not_censored_indices]
            # Sorting out all the non valid binary data:
            relevant_binary_targets = np.reshape(relevant_binary_targets, (relevant_binary_targets.shape[0]))
            valid_binary_target_indices = np.where(relevant_binary_targets != -1)
            relevant_binary_targets = relevant_binary_targets[valid_binary_target_indices]
            relevant_outputs = relevant_outputs[valid_binary_target_indices]

            if len(relevant_binary_targets) != 0:
                fpr_test, tpr_test, _ = roc_curve(relevant_binary_targets, - relevant_outputs)
                roc_auc_test = auc(fpr_test, tpr_test)
            else:
                print('Zero elements')

            test_acc, balanced_acc_test = 0, 0

            print('Performance over Validation Set: Tile AUC: {:.2f}, C index: {} '.format(roc_auc_test, c_index_test))

        else:
            test_acc = 100 * correct_labeling_test / total_test
            balanced_acc_test = 100. * ((true_pos_test + EPS) / (total_pos_test + EPS) + (true_neg_test + EPS) / (total_neg_test + EPS)) / 2

            roc_auc_test = np.nan
            if not all(true_targets_test == true_targets_test[0]):  # more than one label
                fpr_test, tpr_test, _ = roc_curve(true_targets_test, scores_test)
                roc_auc_test = auc(fpr_test, tpr_test)
            all_writer.add_scalar('Test/Balanced Accuracy', balanced_acc_test, epoch)
            all_writer.add_scalar('Test/Accuracy', test_acc, epoch)

            # compute slide AUC  using slide mean score
            patch_df = pd.DataFrame({'slide': slide_names, 'scores': scores_test, 'labels': true_targets_test})
            slide_mean_score_df = patch_df.groupby('slide').mean()
            roc_auc_slide = np.nan
            if not all(slide_mean_score_df['labels'] == slide_mean_score_df['labels'][0]): #more than one label
                roc_auc_slide = roc_auc_score(slide_mean_score_df['labels'], slide_mean_score_df['scores'])
            ############################################################################################################
            if args.n_patches_test > 1:
                all_writer.add_scalar('Test/slide AUC', roc_auc_slide, epoch)
                print('Slide AUC of {:.2f} over Test set'.format(roc_auc_slide))

        all_writer.add_scalar('Test/C index', c_index_test, epoch)
        all_writer.add_scalar('Test/Roc-Auc', roc_auc_test, epoch)


    model.train()
    return test_acc, balanced_acc_test, roc_auc_test




DEVICE = utils.device_gpu_cpu()
sampler = None

# Get data:
if args.target in ['Survival_Time', 'Survival_Binary']:
    train_dset = {'Censored': None,
                  'UnCensored': None}
    test_dset = {'Censored': None,
                 'UnCensored': None}
    batch_size = {'Censored': int(np.ceil(args.batch_size * 0.25)),
                  'UnCensored': int(args.batch_size - np.ceil(args.batch_size * 0.25))}
    train_loader = {'Censored': None,
                    'UnCensored': None}
    test_loader = {'Censored': None,
                   'UnCensored': None}

    for key in train_dset.keys():
        censor_status = True if key == 'Censored' else False
        print('Creating {} Train set'.format(key))
        train_dset[key] = datasets.WSI_REGdataset(DataSet=args.dataset,
                                                  tile_size=TILE_SIZE,
                                                  target_kind=args.target,
                                                  test_fold=args.test_fold,
                                                  train=True,
                                                  print_timing=args.time,
                                                  transform_type=args.transform_type,
                                                  n_tiles=args.n_patches_train,
                                                  color_param=args.c_param,
                                                  get_images=args.images,
                                                  desired_slide_magnification=args.mag,
                                                  DX=args.dx,
                                                  loan=args.loan,
                                                  er_eq_pr=args.er_eq_pr,
                                                  slide_per_block=args.slide_per_block,
                                                  is_Censored=censor_status
                                                  )

        print('Creating {} Test set'.format(key))
        test_dset[key] = datasets.WSI_REGdataset(DataSet=args.dataset,
                                                 tile_size=TILE_SIZE,
                                                 target_kind=args.target,
                                                 test_fold=args.test_fold,
                                                 train=False,
                                                 print_timing=False,
                                                 transform_type='none',
                                                 n_tiles=args.n_patches_test,
                                                 get_images=args.images,
                                                 desired_slide_magnification=args.mag,
                                                 DX=args.dx,
                                                 loan=args.loan,
                                                 er_eq_pr=args.er_eq_pr,
                                                 is_Censored=censor_status
                                                 )





        train_loader[key] = DataLoader(train_dset[key], batch_size=batch_size[key], shuffle=True,
                                       num_workers=0, pin_memory=True, sampler=sampler)
        test_loader[key] = DataLoader(test_dset[key], batch_size=batch_size[key] * 2, shuffle=False,
                                      num_workers=0, pin_memory=True)


    '''if len(train_loader['Censored']) < len(train_loader['UnCensored']):
        print('There are less Censored samples in the train set')
        train_loader = zip(cycle(train_loader['Censored']), train_loader['UnCensored'])
        test_loader = zip(cycle(test_loader['Censored']), test_loader['UnCensored'])
    else:
        print('There are less UnCensored samples in the train set')
        train_loader = zip(train_loader['Censored'], cycle(train_loader['UnCensored']))
        test_loader = zip(test_loader['Censored'], cycle(test_loader['UnCensored']))'''



model = eval(args.model)
if args.target == 'Survival_Time':
    model.change_num_classes(num_classes=1)  # This will convert the liner (classifier) layer into the beta layer
    model.model_name += '_Continous_Time'

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# Choosing the loss function:
if args.target == 'Survival_Time':
    criterion = Cox_loss
else:
    criterion = nn.CrossEntropyLoss()

train(model, train_loader, test_loader, DEVICE=DEVICE, optimizer=optimizer, print_timing=args.time)

print('Finished training')