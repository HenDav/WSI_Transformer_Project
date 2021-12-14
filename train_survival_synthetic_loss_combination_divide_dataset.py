from sksurv.metrics import concordance_index_censored
from datasets import C_Index_Test_Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
from Cox_Loss import Combined_loss
import torch.optim as optim
import utils
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import argparse
from sklearn.metrics import roc_curve, auc
import os
from pathlib import Path
from itertools import cycle
from random import shuffle

parser = argparse.ArgumentParser(description='')
parser.add_argument('-e', '--epochs', default=500, type=int, help='Epochs to run')
parser.add_argument('-tar', '--target', type=str, default='Time', help='Binary / Time')
parser.add_argument('-l', '--loss', type=str, default='L2', help='Cox / L2')
parser.add_argument('-fe', '--from_epoch', type=int, default=0, help='Continue train from epoch')
parser.add_argument('-mb', '--mini_batch_size', type=int, default=20, help='Mini batch size')
parser.add_argument('-tm', '--train_mode', type=str, default='T2T', help='B2B, B2T, T2B, T2T')
parser.add_argument('-wc', '--without_censored', dest='without_censored', action='store_true', help='train without censpred data')
parser.add_argument('--lr', default=40e-5, type=float, help='learning rate')
parser.add_argument('-lw', '--loss_weights', type=float, default=[0, 0, 1], help='loss weights')
parser.add_argument('-cr', '--censored_ratio', type=float, default=0.5, help='ratio of not censored samples in each minibatch')
args = parser.parse_args()

# Calculating the number of censored and not censored samples in each minibatch:
if args.censored_ratio > 0 and args.censored_ratio < 1:
    args.not_censored = int(np.ceil((args.censored_ratio * args.mini_batch_size)))
    args.censored = args.mini_batch_size - args.not_censored
elif args.censored_ratio == 1:  # All samples will be not censored
    args.not_censored = args.mini_batch_size
    args.censored = 0
elif args.censored_ratio == 0:
    raise Exception('The value of censored_ratio CANNOT be grater than 0 since all samples will be censored and that is unacceptable')
elif args.censored_ratio > 1:
    raise Exception('The value of censored_ratio CANNOT be grater than 1')
elif args.censored_ratio < 0:
    print('Dataset will be initiated as is (without any manipulations of censored/ not censored number of samples in the minibatch)')


parameters = [-1.131956611, -0.156472727, -0.575693902, 0.495659331, -0.514212918, 0.426040546, -0.578776545, -0.755408599]
bias = 5.82919158508615

def train(from_epoch: int = 0, epochs: int = 2, data_loader = None):
    minibatch_len = len(data_loader['Not Censored']) if type(data_loader['Censored']) == cycle else len(data_loader['Censored'])
    for e in tqdm(range(from_epoch, from_epoch + epochs)):
        dset_size_small, dset_size_large = 0, 0

        all_target_time, all_target_binary, all_censored = [], [], []
        all_outputs_Cox, all_outputs_L2, all_outputs_Binary = [], [], []

        train_loss, loss_cox_total, loss_L2_total, loss_cross_entropy_total = 0, 0, 0, 0
        dLoss__d_outputs_cox, dLoss__d_outputs_L2, dLoss__d_outputs_cross_entropy = 0, 0, 0
        dLoss__d_outputs = 0

        for batch_idx, minibatch in enumerate(zip(cycle(data_loader['Censored']), data_loader['Not Censored'])):
            dset_size_small += len(minibatch[0]['Censored'])
            dset_size_large += len(minibatch[1]['Censored'])
            time_stamp = batch_idx + e * minibatch_len
            if minibatch[1] == 0:  # There is 1 combined dataset that includes all samples
                minibatch = minibatch[0]

            else:  # There are 2 dataset, one for Censored samples and one for not censored samples. Those 2 datasets need to be combined
                # The data in the combined dataset is divided into censored and not censored.
                #  We'll shuffle it:
                indices = list(range(minibatch[0]['Censored'].size(0) + minibatch[1]['Censored'].size(0)))
                shuffle(indices)

                temp_minibatch = {}
                for key in minibatch[0].keys():
                    if type(minibatch[0][key]) == torch.Tensor:
                        temp_minibatch[key] = torch.cat([minibatch[0][key], minibatch[1][key]], dim=0)[indices]
                    else:
                        raise Exception('Could not find the type for this data')

                minibatch = temp_minibatch

            data = minibatch['Features']
            target_time = minibatch['Time Target']
            target_binary = minibatch['Binary Target']
            censored = minibatch['Censored']

            censored_ratio = len(np.where(censored == True)[0]) / len(censored)
            all_writer.add_scalar('Train/Censored Ratio per Minibatch', censored_ratio, time_stamp)
            if censored_ratio != args.censored_ratio:
                print('Computed ratio is {} Vs. designed ratio {}. Minibatch {}/{}'.format(censored_ratio, args.censored_ratio,
                                                                                           batch_idx, minibatch_len))
                print('Small {}. Large {}'.format(dset_size_small, dset_size_large))
                print()

            all_target_time.extend(target_time.numpy())
            all_target_binary.extend(target_binary.numpy())
            all_censored.extend(censored.numpy())

            '''
            if args.target == 'Time':
                all_target_time.extend(target_time.numpy())
                all_target_binary.extend(target_binary.numpy())
            elif args.target == 'Binary':
                # if we're using the binary model than we'll take only the data whose targets are not -1:
                valid_indices = target_binary != -1
                data = data[valid_indices]
                target_binary = target_binary[valid_indices]
                target_time = target_time[valid_indices]
                censored = censored[valid_indices]
                all_target_binary.extend(target_binary.numpy())
                all_target_time.extend(target_time.numpy())
            
            all_censored.extend(censored.numpy())
            '''

            data = data.to(DEVICE)
            target_binary = target_binary.to(DEVICE)
            target_time = target_time.to(DEVICE)

            model.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(data)

            all_outputs_Cox.extend(outputs.detach().cpu().numpy()[:, 0])
            all_outputs_L2.extend(outputs.detach().cpu().numpy()[:, 1])
            outputs_for_binary = torch.nn.functional.softmax(outputs[:, 2:], dim=1)
            all_outputs_Binary.extend(outputs_for_binary[:, 1].detach().cpu().numpy())
            outputs.retain_grad()  # FIXME: checking how to retrieve gradients

            loss, loss_cox, loss_L2, loss_cross_entropy = criterion(outputs, targets_time=target_time, targets_binary=target_binary, censored=censored, weights=args.loss_weights)

            '''
            if args.target == 'Time':
                all_outputs.extend(outputs.detach().cpu().numpy()[:, 0])
                loss = criterion(outputs, target_time, censored)

            elif args.target == 'Binary':
                outputs = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                all_outputs.extend(outputs[:, 1].detach().cpu().numpy())
                loss = criterion(outputs, target_binary)
            '''

            # loss.register_hook(lambda grad: print(grad))  # FIXME: checking how to retrieve gradients
            # model.weight.register_hook(lambda grad: print(grad))  # FIXME: checking how to retrieve gradients
            # model.bias.register_hook(lambda grad: print(grad))  # FIXME: checking how to retrieve gradients

            loss.backward()
            dLoss__d_outputs_cox += np.sum(np.abs(outputs.grad.detach().cpu().numpy()[:, 0]))
            dLoss__d_outputs_L2 += np.sum(np.abs(outputs.grad.detach().cpu().numpy()[:, 1]))
            dLoss__d_outputs_cross_entropy += np.sum(np.abs(outputs.grad.detach().cpu().numpy()[:, 2]))
            dLoss__d_outputs += np.sum(np.abs(outputs.grad.detach().cpu().numpy()))

            optimizer.step()
            train_loss += loss.item()
            loss_cox_total += loss_cox.item()
            loss_L2_total += loss_L2.item()
            loss_cross_entropy_total += loss_cross_entropy.item()

            all_writer.add_scalar('Train/Loss per Minibatch', loss, time_stamp)

        # Compute C index:
        # PAY ATTENTION: the function 'concordance_index_censored' takes censored = True as not censored (This is why we should invert 'all_censored' )
        # and takes the outputs as a risk NOT as survival time !!!!!!!!!!
        #all_outputs_for_c_index = -np.array(all_outputs) if flip_outputs else all_outputs
        c_index_Cox, _, _, _, _ = concordance_index_censored(np.invert(all_censored),
                                                             all_target_time,
                                                             all_outputs_Cox)
        c_index_L2, _, _, _, _ = concordance_index_censored(np.invert(all_censored),
                                                            all_target_time,
                                                            -np.array(all_outputs_L2))
        c_index_Binary, _, _, _, _ = concordance_index_censored(np.invert(all_censored),
                                                                all_target_time,
                                                                -np.array(all_outputs_Binary))

        # Compute AUC according to Cox model and L2:
        # When computing AUC we need binary data. We don't care if a patient is censored or not as long as it has
        # A binary target. So, sorting out all the censored patients is not necessary.
        # For computing AUC we just need to get rid of all the non targeted samples
        # Sorting out all the non valid binary data:
        valid_binary_target_indices = np.where(np.array(all_target_binary) != -1)[0]
        relevant_binary_targets = np.array(all_target_binary)[valid_binary_target_indices]

        # When using Cox model the scores represent Risk so we should invert them for computing AUC
        all_outputs_Cox = -np.array(all_outputs_Cox)
        relevant_outputs_Cox = np.array(all_outputs_Cox)[valid_binary_target_indices]
        relevant_outputs_L2 = np.array(all_outputs_L2)[valid_binary_target_indices]

        fpr_train_Cox, tpr_train_Cox, _ = roc_curve(relevant_binary_targets, relevant_outputs_Cox)
        roc_auc_train_Cox = auc(fpr_train_Cox, tpr_train_Cox)

        fpr_train_L2, tpr_train_L2, _ = roc_curve(relevant_binary_targets, relevant_outputs_L2)
        roc_auc_train_L2 = auc(fpr_train_L2, tpr_train_L2)

        # When working with the Binary model than all data without binary target should be sorted out.
        fpr_train_Binary, tpr_train_Binary, _ = roc_curve(relevant_binary_targets, np.array(all_outputs_Binary)[valid_binary_target_indices])
        roc_auc_train_Binary = auc(fpr_train_Binary, tpr_train_Binary)

        #all_writer.add_scalar('Train/Loss Per Epoch', train_loss, e)
        #all_writer.add_scalar('Train/C-index Per Epoch', c_index, e)
        #all_writer.add_scalar('Train/AUC Per Epoch', roc_auc_train, e)
        #all_writer.add_scalar('Train/d_Loss/d_outputs Per Epoch', dLoss__d_outputs, e)
        #all_writer.add_scalar('Train/Gradient Cox', dLoss__d_outputs_cox, e)
        #all_writer.add_scalar('Train/Gradient L2', dLoss__d_outputs_L2, e)
        #all_writer.add_scalar('Train/Gradient Binary', dLoss__d_outputs_cox, e)

        all_writer.add_scalars('Train/Loss Per Epoch', {'Total': train_loss,
                                                        'Cox': loss_cox_total,
                                                        'L2': loss_L2_total,
                                                        'Binary': loss_cross_entropy_total
                                                        }, e)

        all_writer.add_scalars('Train/Gradients', {'Total': dLoss__d_outputs,
                                                   'Cox': dLoss__d_outputs_cox,
                                                   'L2': dLoss__d_outputs_L2,
                                                   'Binary': dLoss__d_outputs_cross_entropy
                                                   }, e)

        all_writer.add_scalars('Train/C-index', {'Cox': c_index_Cox,
                                                 'L2': c_index_L2,
                                                 'Binary': c_index_Binary
                                                 }, e)

        all_writer.add_scalars('Train/AUC', {'Cox': roc_auc_train_Cox,
                                             'L2': roc_auc_train_L2,
                                             'Binary': roc_auc_train_Binary
                                             }, e)

        if e % 20 == 0:
            print('Loss: {},  Mean C-index: {}, Mean AUC: {}'.format(train_loss,
                                                                     (c_index_Cox + c_index_L2 + c_index_Binary) / 3,
                                                                     (roc_auc_train_Cox + roc_auc_train_L2 + roc_auc_train_Binary) / 3))
            # Run validation:
            test(e, test_loader)

            # Save model to file:
            try:
                model_state_dict = model.module.state_dict()
            except AttributeError:
                model_state_dict = model.state_dict()
            torch.save({'epoch': e,
                        'model_state_dict': model_state_dict
                        },
                       os.path.join(main_dir, 'Model_CheckPoints', 'model_data_Epoch_' + str(e) + '.pt'))


    all_writer.close()


def test(current_epoch, test_data_loader):
    model.eval()
    model.to(DEVICE)

    all_target_time, all_target_binary, all_censored = [], [], []
    all_outputs_Cox, all_outputs_L2, all_outputs_Binary = [], [], []
    test_loss = 0

    with torch.no_grad():
        for batch_idx, minibatch in enumerate(zip(test_data_loader['Censored'], test_data_loader['Not Censored'])):
            if minibatch[1] == 0:  # There is 1 combined dataset that includes all samples
                minibatch = minibatch[0]

            else:  # There are 2 dataset, one for Censored samples and one for not censored samples. Those 2 datasets need to be combined
                # There is no need to shuffle the samples for the validation set
                temp_minibatch = {}
                for key in minibatch[0].keys():
                    if type(minibatch[0][key]) == torch.Tensor:
                        temp_minibatch[key] = torch.cat([minibatch[0][key], minibatch[1][key]], dim=0)
                    else:
                        raise Exception('Could not find the type for this data')

                minibatch = temp_minibatch

            data = minibatch['Features']
            target_time = minibatch['Time Target']
            target_binary = minibatch['Binary Target']
            censored = minibatch['Censored']

            all_target_time.extend(target_time.numpy())
            all_target_binary.extend(target_binary.numpy())
            all_censored.extend(censored.numpy())

            '''
            if args.target == 'Time':
                all_targets_time.extend(target_time.numpy())
                all_targets_binary.extend(target_binary.numpy())
            elif args.target == 'Binary':
                # if we're using the binary model than we'll take only the datat whose targets are not -1:
                valid_indices = target_binary != -1
                data = data[valid_indices]
                target_binary = target_binary[valid_indices]
                target_time = target_time[valid_indices]
                censored = censored[valid_indices]
                all_targets_time.extend(target_time.numpy())
                all_targets_binary.extend(target_binary.numpy())

            all_censored.extend(censored.numpy())
            '''

            data = data.to(DEVICE)
            target_binary = target_binary.to(DEVICE)
            target_time = target_time.to(DEVICE)

            outputs = model(data)

            all_outputs_Cox.extend(outputs.detach().cpu().numpy()[:, 0])
            all_outputs_L2.extend(outputs.detach().cpu().numpy()[:, 1])
            outputs_for_binary = torch.nn.functional.softmax(outputs[:, 2:], dim=1)
            all_outputs_Binary.extend(outputs_for_binary[:, 1].detach().cpu().numpy())

            loss, _, _, _ = criterion(outputs, targets_time=target_time, targets_binary=target_binary, censored=censored, weights=args.loss_weights)
            test_loss += loss.item()

            '''
            if args.target == 'Time':
                all_outputs.extend(outputs.detach().cpu().numpy()[:,0])
                loss = criterion(outputs, target_time, censored)
            elif args.target == 'Binary':
                outputs = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                all_outputs.extend(outputs[:, 1].detach().cpu().numpy())
                loss = criterion(outputs, target_binary)
            '''

    # Compute C index:
    #all_outputs_for_c_index = -np.array(all_outputs) if flip_outputs else all_outputs
    c_index_Cox, _, _, _, _ = concordance_index_censored(np.invert(all_censored),
                                                         all_target_time,
                                                         all_outputs_Cox)
    c_index_L2, _, _, _, _ = concordance_index_censored(np.invert(all_censored),
                                                        all_target_time,
                                                        -np.array(all_outputs_L2))
    c_index_Binary, _, _, _, _ = concordance_index_censored(np.invert(all_censored),
                                                            all_target_time,
                                                            -np.array(all_outputs_Binary))

    # Compute AUC:
    valid_binary_target_indices = np.where(np.array(all_target_binary) != -1)[0]
    relevant_binary_targets = np.array(all_target_binary)[valid_binary_target_indices]

    # When using Cox model the scores represent Risk so we should invert them for computing AUC
    all_outputs_Cox = -np.array(all_outputs_Cox)
    relevant_outputs_Cox = np.array(all_outputs_Cox)[valid_binary_target_indices]
    relevant_outputs_L2 = np.array(all_outputs_L2)[valid_binary_target_indices]

    fpr_test_Cox, tpr_test_Cox, _ = roc_curve(relevant_binary_targets, relevant_outputs_Cox)
    auc_test_Cox = auc(fpr_test_Cox, tpr_test_Cox)

    fpr_test_L2, tpr_test_L2, _ = roc_curve(relevant_binary_targets, relevant_outputs_L2)
    auc_test_L2 = auc(fpr_test_L2, tpr_test_L2)

    fpr_test_Binary, tpr_test_Binary, _ = roc_curve(relevant_binary_targets, np.array(all_outputs_Binary)[valid_binary_target_indices])
    auc_test_Binary = auc(fpr_test_Binary, tpr_test_Binary)


    print('Validation set Performance. Mean C-index: {}, Mean AUC: {}'.format((c_index_Cox + c_index_L2 + c_index_Binary) / 3,
                                                                              (auc_test_Cox + auc_test_L2 + auc_test_Binary) / 3))
    all_writer.add_scalar('Test/Loss Per Epoch', test_loss, current_epoch)
    #all_writer.add_scalar('Test/C-index Per Epoch', c_index, current_epoch)
    #all_writer.add_scalar('Test/AUC Per Epoch', roc_auc_test, current_epoch)

    all_writer.add_scalars('Test/C-index', {'Cox': c_index_Cox,
                                            'L2': c_index_L2,
                                            'Binary': c_index_Binary
                                            }, current_epoch)

    all_writer.add_scalars('Test/AUC', {'Cox': auc_test_Cox,
                                        'L2': auc_test_L2,
                                        'Binary': auc_test_Binary
                                        }, current_epoch)

    model.train()

########################################################################################################################
########################################################################################################################


#args.without_censored = True

# Model definition:
# The model be consist a 1st FC layer from 8 -> 4
# and a 2nd layer from 4 -> 1/1/2 (outputs)
# Each output is intended to be used for a different kind of target.
# 1. Risk calaulation (1 score) using Cox loss
# 2. Survival time calaulation (1 score) using L2 Norm loss
# 3. Binary Survival (5 Yrs.) (2 scores) using Cross entropy loss

model = nn.Sequential(nn.Linear(8, 2),
                      nn.Linear(2, 4)
                      )
# main_dir = 'Test_Run/Loss_Combination_Step_' + str(args.lr) + '_Censored_' + '_'.join(str(args.censored_ratio).split('.')) + '_Weights_' + '_'.join([str(num) for num in args.loss_weights])
main_dir = 'Test_Run/Loss_Combination_Step_' + str(args.lr) + '_Censored_' + str(args.censored_ratio) + '_Weights_' + '_'.join([str(num) for num in args.loss_weights])
criterion = Combined_loss

main_dir = main_dir + '_without_Censored' if args.without_censored else main_dir

# Continue train from previous epoch:
if args.from_epoch != 0:
    # Load model:
    print('Loading pre-saved model...')
    if args.train_mode in ['B2B', 'B2T']:  # Load binary model
        model_dir = 'Test_Run/Binary/Model_CheckPoints'
        main_dir = 'Test_Run/B2T' if args.train_mode == 'B2T' else main_dir

    elif args.train_mode in ['T2B', 'T2T']:  # Load Time model
        model_dir = 'Test_Run/Time_L2/Model_CheckPoints' if args.loss == 'L2' else 'Test_Run/Time_Cox/Model_CheckPoints'
        main_dir = 'Test_Run/T2B' if args.train_mode == 'T2B' else main_dir

    model_data_loaded = torch.load(os.path.join(model_dir,
                                                'model_data_Epoch_' + str(args.from_epoch) + '.pt'),
                                   map_location='cpu')

    from_epoch = args.from_epoch + 1

    if args.train_mode == 'B2T':
        parameters = model_data_loaded['model_state_dict']['weight'][1, :] - model_data_loaded['model_state_dict']['weight'][0, :]
        bias = model_data_loaded['model_state_dict']['bias'][1] - model_data_loaded['model_state_dict']['bias'][0]

        model.weight.data[0, :] = parameters
        model.bias.data = bias
else:
    from_epoch = 0

check_parameters = False
check_optimization = False
if check_parameters or check_optimization:
    model.bias.data = torch.ones(1) * bias
    for digit_num in range(model.weight.data.size(1)):
        model.weight.data[0][digit_num] = torch.ones(1) * parameters[digit_num]

    if check_parameters:
        model.eval()
    elif check_optimization:
        optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=5e-5)
        model.train()
else:
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-5)
    model.train()

DEVICE = utils.device_gpu_cpu()

if args.censored_ratio < 0:  # We're using the data as is and not manipulating the amount of censored/not censored samples in each minibatch
    train_set = C_Index_Test_Dataset(train=True)
    test_set = C_Index_Test_Dataset(train=False)

elif args.censored_ratio > 0 and args.censored_ratio < 1:
    train_set_censored = C_Index_Test_Dataset(train=True, is_all_censored=True)
    train_set_not_censored = C_Index_Test_Dataset(train=True, is_all_not_censored=True)

    test_set_censored = C_Index_Test_Dataset(train=False, is_all_censored=True)
    test_set_not_censored = C_Index_Test_Dataset(train=False, is_all_not_censored=True)

elif args.censored_ratio == 1:
    train_set = C_Index_Test_Dataset(train=True, is_all_censored=True)
    test_set = C_Index_Test_Dataset(train=False, is_all_censored=True)


if args.censored_ratio < 0 or args.censored_ratio == 1:
    train_loader = {'Censored': DataLoader(train_set, batch_size=args.mini_batch_size, shuffle=True),
                    'Not Censored': cycle([0])}
    test_loader = {'Censored': DataLoader(test_set, batch_size=args.mini_batch_size, shuffle=False),
                   'Not Censored': cycle([0])}

elif args.censored_ratio > 0 and args.censored_ratio < 1:
    train_loader_censored = DataLoader(train_set_censored, batch_size=args.censored, shuffle=True, drop_last=True)
    train_loader_not_censored = DataLoader(train_set_not_censored, batch_size=args.not_censored, shuffle=True, drop_last=True)
    '''if len(train_loader_censored) < len(train_loader_not_censored):
        small_loader_len = len(train_loader_censored)  # FIXME: Remove after debugging
        train_loader_censored = cycle(train_loader_censored)
    elif len(train_loader_censored) > len(train_loader_not_censored):
        small_loader_len = len(train_loader_not_censored)  # FIXME: Remove after debugging
        train_loader_not_censored = cycle(train_loader_not_censored)'''

    train_loader = {'Censored': train_loader_censored,
                    'Not Censored': train_loader_not_censored
                    }

    test_loader_censored = DataLoader(test_set_censored, batch_size=args.censored, shuffle=False)
    test_loader_not_censored = DataLoader(test_set_not_censored, batch_size=args.not_censored, shuffle=False)
    if len(test_loader_censored) < len(test_loader_not_censored):
        test_loader_censored = cycle(test_loader_censored)
    elif len(test_loader_censored) > len(test_loader_not_censored):
        test_loader_not_censored = cycle(test_loader_not_censored)

    test_loader = {'Censored': test_loader_censored,
                   'Not Censored': test_loader_not_censored
                   }

all_writer = SummaryWriter(main_dir)

if not os.path.isdir(os.path.join(main_dir, 'Model_CheckPoints')):
    Path(os.path.join(main_dir, 'Model_CheckPoints')).mkdir(parents=True)

if check_parameters:
    test(current_epoch=1, test_data_loader=train_loader)
else:
    train(from_epoch=from_epoch, epochs=args.epochs, data_loader=train_loader)

print('Done')

