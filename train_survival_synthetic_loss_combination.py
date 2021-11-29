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

parser = argparse.ArgumentParser(description='')
parser.add_argument('-e', '--epochs', default=1500, type=int, help='Epochs to run')
parser.add_argument('-tar', '--target', type=str, default='Time', help='Binary / Time')
parser.add_argument('-l', '--loss', type=str, default='L2', help='Cox / L2')
parser.add_argument('-fe', '--from_epoch', type=int, default=0, help='Continue train from epoch')
parser.add_argument('-mb', '--mini_batch_size', type=int, default=20, help='Mini batch size')
parser.add_argument('-tm', '--train_mode', type=str, default='T2T', help='B2B, B2T, T2B, T2T')
parser.add_argument('-wc', '--without_censored', dest='without_censored', action='store_true', help='train without censpred data')
parser.add_argument('--lr', default=20e-5, type=float, help='learning rate')
parser.add_argument('-lw', '--loss_weights', type=int, default=[1, 1, 1], help='loss weights')
args = parser.parse_args()


parameters = [-1.131956611, -0.156472727, -0.575693902, 0.495659331, -0.514212918, 0.426040546, -0.578776545, -0.755408599]
bias = 5.82919158508615

def train(from_epoch: int = 0, epochs: int = 2, data_loader = None):
    for e in tqdm(range(from_epoch, from_epoch + epochs)):

        all_target_time, all_target_binary, all_censored = [], [], []
        all_outputs_Cox, all_outputs_L2, all_outputs_Binary = [], [], []

        train_loss = 0
        dLoss__d_outputs_cox, dLoss__d_outputs_L2, dLoss__d_outputs_cross_entropy = 0, 0, 0
        dLoss__d_outputs = 0

        for batch_idx, minibatch in enumerate(data_loader):
            time_stamp = batch_idx + e * len(data_loader)
            data = minibatch['Features']
            target_time = minibatch['Time Target']
            target_binary = minibatch['Binary Target']
            censored = minibatch['Censored']

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

            loss = criterion(outputs, targets_time=target_time, targets_binary=target_binary, censored=censored, weights=args.loss_weights)

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

            all_writer.add_scalar('Train/Loss per Minibatch', loss, time_stamp)

        # Compute C index:
        # PAY ATTENTION: the function 'concordance_index_censored' takes censored = True as not censored (This is why we should invert 'all_censored' )
        # and takes the outputs as a risk NOT as survival time !!!!!!!!!!
        #all_outputs_for_c_index = -np.array(all_outputs) if flip_outputs else all_outputs
        c_index_Cox, _, _, _, _ = concordance_index_censored(np.invert(all_censored),
                                                             all_target_time,
                                                             all_outputs
                                                             )
        c_index_L2, _, _, _, _ = concordance_index_censored(np.invert(all_censored),
                                                            all_target_time,
                                                            -np.array(all_outputs)
                                                            )
        c_index_Binary, _, _, _, _ = concordance_index_censored(np.invert(all_censored),
                                                                all_target_time,
                                                                -np.array(all_outputs)
                                                                )

        # Compute AUC according to Cox model and L2:
        # When computing AUC we need binary data. We don't care if a patient is censored or not as long as it has
        # A binary target. So, sorting out all the censored patients is not necessary.
        # For computing AUC we just need to get rid of all the non targeted samples
        # Sorting out all the non valid binary data:
        valid_binary_target_indices = np.where(np.array(all_target_binary) != -1)[0]
        relevant_binary_targets = np.array(all_target_binary)[valid_binary_target_indices]

        # When using Cox model the scores represent Risk so we should invert them for computing AUC
        all_outputs_Cox = -np.array(all_outputs)
        relevant_outputs_Cox = np.array(all_outputs_Cox)[valid_binary_target_indices]
        relevant_outputs_L2 = np.array(all_outputs)[valid_binary_target_indices]

        fpr_train_Cox, tpr_train_Cox, _ = roc_curve(relevant_binary_targets, relevant_outputs_Cox)
        roc_auc_train_Cox = auc(fpr_train_Cox, tpr_train_Cox)

        fpr_train_L2, tpr_train_L2, _ = roc_curve(relevant_binary_targets, relevant_outputs_L2)
        roc_auc_train_L2 = auc(fpr_train_L2, tpr_train_L2)

        # When working with the Binary model than all data without binary target should be sorted out.

        fpr_train_Binary, tpr_train_Binary, _ = roc_curve(all_target_binary, all_outputs)
        roc_auc_train_Binary = auc(fpr_train_Binary, tpr_train_Binary)

        all_writer.add_scalar('Train/Loss Per Epoch', train_loss, e)
        all_writer.add_scalar('Train/C-index Per Epoch', c_index, e)
        all_writer.add_scalar('Train/AUC Per Epoch', roc_auc_train, e)
        all_writer.add_scalar('Train/d_Loss/d_outputs Per Epoch', dLoss__d_outputs, e)

        all_writer.add_scalars('Train/Gradients', {'Total': dLoss__d_outputs,
                                                   'Cox': dLoss__d_outputs_cox,
                                                   'L2': dLoss__d_outputs_L2,
                                                   'Binary': dLoss__d_outputs_cross_entropy
                                                   }, e)

        if e % 20 == 0:
            print('Loss: {}, C-index: {}, AUC: {}'.format(train_loss, c_index, roc_auc_train))
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

    all_targets_time, all_targets_binary, all_outputs, all_censored = [], [], [], []
    test_loss = 0

    with torch.no_grad():
        for batch_idx, minibatch in enumerate(test_data_loader):
            data = minibatch['Features']
            target_time = minibatch['Time Target']
            target_binary = minibatch['Binary Target']
            censored = minibatch['Censored']
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
            data = data.to(DEVICE)
            target_time = target_time.to(DEVICE)
            target_binary = target_binary.to(DEVICE)

            outputs = model(data)

            if args.target == 'Time':
                all_outputs.extend(outputs.detach().cpu().numpy()[:,0])
                loss = criterion(outputs, target_time, censored)
            elif args.target == 'Binary':
                outputs = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                all_outputs.extend(outputs[:, 1].detach().cpu().numpy())
                loss = criterion(outputs, target_binary)

            test_loss += loss.item()

        # Compute C index:
        all_outputs_for_c_index = -np.array(all_outputs) if flip_outputs else all_outputs
        c_index, num_concordant_pairs, num_discordant_pairs, _, _ = concordance_index_censored(np.invert(all_censored),
                                                                                               all_targets_time,
                                                                                               all_outputs_for_c_index
                                                                                               )
        # Compute AUC:
        if args.target == 'Time':
            valid_binary_target_indices = np.where(np.array(all_targets_binary) != -1)[0]
            relevant_binary_targets = np.array(all_targets_binary)[valid_binary_target_indices]

            # When using Cox model the scores represent Risk so we should invert them for computing AUC
            all_outputs = -np.array(all_outputs) if (args.target == 'Time' and args.loss == 'Cox') else all_outputs
            relevant_outputs = np.array(all_outputs)[valid_binary_target_indices]

            fpr_test, tpr_test, _ = roc_curve(relevant_binary_targets, relevant_outputs)
            roc_auc_test = auc(fpr_test, tpr_test)

        elif args.target == 'Binary':
            fpr_test, tpr_test, _ = roc_curve(all_targets_binary, all_outputs)
            roc_auc_test = auc(fpr_test, tpr_test)


        print('Validation set Performance. C-index: {}, AUC: {}'.format(c_index, roc_auc_test))
        all_writer.add_scalar('Test/Loss Per Epoch', test_loss, current_epoch)
        all_writer.add_scalar('Test/C-index Per Epoch', c_index, current_epoch)
        all_writer.add_scalar('Test/AUC Per Epoch', roc_auc_test, current_epoch)

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

model = nn.Sequential(nn.Linear(8, 4),
                      nn.Linear(4, 4)
                      )
main_dir = 'Test_Run/Loss_Combination'
criterion = Combined_loss
'''
if args.target == 'Time':
    model = nn.Linear(8, 1)
    if args.loss == 'Cox':
        criterion = Cox_loss
        flip_outputs = False
        main_dir = 'Test_Run/Time_Cox'
    elif args.loss == 'L2':
        criterion = L2_Loss
        flip_outputs = True
        main_dir = 'Test_Run/Time_L2'
    else:
        Exception('No valid loss function defined')

elif args.target == 'Binary':
    model = nn.Linear(8, 2)
    criterion = nn.CrossEntropyLoss()
    flip_outputs = True
    main_dir = 'Test_Run/Binary'
'''

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

train_set = C_Index_Test_Dataset(train=True, without_censored=args.without_censored)
test_set = C_Index_Test_Dataset(train=False, without_censored=args.without_censored)

train_loader = DataLoader(train_set, batch_size=args.mini_batch_size, shuffle=True)
test_loader = DataLoader(train_set, batch_size=args.mini_batch_size, shuffle=False)


all_writer = SummaryWriter(main_dir)

if not os.path.isdir(os.path.join(main_dir, 'Model_CheckPoints')):
    Path(os.path.join(main_dir, 'Model_CheckPoints')).mkdir(parents=True)

if check_parameters:
    test(current_epoch=1, test_data_loader=train_loader)
else:
    train(from_epoch=from_epoch, epochs=args.epochs, data_loader=train_loader)

print('Done')

