from sksurv.metrics import concordance_index_censored
from datasets import Features_to_Survival
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
from Cox_Loss import Cox_loss, L2_Loss
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
from PreActResNets import PreActResNet50_Ron
import copy

parser = argparse.ArgumentParser(description='')
parser.add_argument('-e', '--epochs', default=501, type=int, help='Epochs to run')
parser.add_argument('-tar', '--target', type=str, default='Time', help='Binary / Time')
parser.add_argument('-l', '--loss', type=str, default='L2', help='Cox / L2')
parser.add_argument('-fe', '--from_epoch', type=int, default=0, help='Continue train from epoch')
parser.add_argument('-mb', '--mini_batch_size', type=int, default=20, help='Mini batch size')
#parser.add_argument('-wc', '--without_censored', dest='without_censored', action='store_true', help='train without censpred data')
parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=5e4, type=float, help='L2 penalty')
parser.add_argument('--eps', default=1e-5, type=float, help='epsilon (for optimizer')
parser.add_argument('-cr', '--censored_ratio', type=float, default=-0.5, help='ratio of censored samples in each minibatch')
args = parser.parse_args()

# Calculating the number of censored and not censored samples in each minibatch:
if args.censored_ratio > 0 and args.censored_ratio < 1:
    args.censored = int(np.ceil((args.censored_ratio * args.mini_batch_size)))
    args.not_censored = args.mini_batch_size - args.censored
elif args.censored_ratio == 0:  # All samples will be of kind "Not Censored"
    args.not_censored = args.mini_batch_size
    args.censored = 0
elif args.censored_ratio == 1:
    raise Exception('The value of censored_ratio CANNOT be 1 since all samples will be censored and that is unacceptable')
elif args.censored_ratio > 1:
    raise Exception('The value of censored_ratio CANNOT be grater than 1')
elif args.censored_ratio < 0:
    print('Dataset will be initiated as is (without any manipulations of censored/ not censored number of samples in the minibatch)')


def train(from_epoch: int = 0, epochs: int = 2, data_loader=None):
    model_parameters = {'Weights': copy.deepcopy(model.state_dict()['weight'].detach().cpu().numpy()),
                        'Bias': copy.deepcopy(model.state_dict()['bias'].detach().cpu().numpy())
                        }

    minibatch_len = len(data_loader['Not Censored']) if type(data_loader['Censored']) == cycle else len(data_loader['Censored'])
    for e in tqdm(range(from_epoch, from_epoch + epochs)):
        all_target_time, all_target_binary, all_outputs, all_censored = [], [], [], []
        train_loss = 0
        dLoss__d_outputs = 0

        samples_per_epoch = 0

        model.train()
        model.to(DEVICE)

        for batch_idx, minibatch in enumerate(zip(data_loader['Censored'], data_loader['Not Censored'])):
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

                    elif type(minibatch[0][key]) == list:
                        minibatch[0][key].extend(minibatch[0][key])

                    else:
                        raise Exception('Could not find the type for this data')

                minibatch = temp_minibatch

            data = minibatch['Features'].squeeze(1)
            target_time = minibatch['Time Target']
            target_binary = minibatch['Binary Target']
            censored = minibatch['Censored']

            samples_per_epoch += len(censored)
            censored_ratio = len(np.where(censored == True)[0]) / len(censored)
            all_writer.add_scalar('Train/Censored Ratio per Minibatch', censored_ratio, time_stamp)

            all_target_time.extend(target_time.numpy())
            all_target_binary.extend(target_binary.numpy())
            all_censored.extend(censored.numpy())

            data = data.to(DEVICE)
            target_binary = target_binary.to(DEVICE)
            target_time = target_time.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(data)
            outputs.retain_grad()  # FIXME: checking how to retrieve gradients

            if args.target == 'Time':
                all_outputs.extend(outputs.detach().cpu().numpy()[:, 0])
                loss = criterion(outputs, target_time, censored)
                if np.isnan(loss.item()):
                    print('Got Nan in Loss computation')


            elif args.target == 'Binary':
                outputs_after_sftmx = torch.nn.functional.softmax(outputs, dim=1)
                all_outputs.extend(outputs_after_sftmx[:, 1].detach().cpu().numpy())

                # For loss computations we need only the relevant outputs and binary targets - the ones that has a valid target (not -1)
                valid_indices = np.where(target_binary != -1)[0]
                outputs = outputs[valid_indices]
                target_binary = target_binary[valid_indices]
                outputs.retain_grad()
                loss = criterion(outputs, target_binary)

            loss.backward()
            dLoss__d_outputs += np.sum(np.abs(outputs.grad.detach().cpu().numpy()))
            optimizer.step()
            train_loss += loss.item()

            new_model_parameters = {'Weights': copy.deepcopy(model.state_dict()['weight'].detach().cpu().numpy()),
                                    'Bias': copy.deepcopy(model.state_dict()['bias'].detach().cpu().numpy())
                                    }
            model_abs_difference = {'Weights': np.abs(new_model_parameters['Weights'] - model_parameters['Weights']),
                                    'Bias': np.abs(new_model_parameters['Bias'] - model_parameters['Bias'])
                                    }
            model_parameters = new_model_parameters
            model_max_diff = max(np.max(model_abs_difference['Weights']), np.max(model_abs_difference['Bias']))
            max_parameter = max(torch.max(model.weight).item(), torch.max(model.bias).item())
            all_writer.add_scalar('Train/Model Parameters Max Difference per Minibatch', model_max_diff, time_stamp)
            all_writer.add_scalar('Train/Model Parameters Max Difference / Max Parameter per Minibatch', model_max_diff / max_parameter, time_stamp)

            all_writer.add_scalar('Train/Loss per Minibatch', loss, time_stamp)

        if 'scheduler' in locals():
            scheduler.step()

        # Compute C index:
        # PAY ATTENTION: the function 'concordance_index_censored' takes censored = True as not censored (This is why we should invert 'all_censored' )
        # and takes the outputs as a risk NOT as survival time !!!!!!!!!!
        all_outputs_for_c_index = -np.array(all_outputs) if flip_outputs else all_outputs
        c_index, num_concordant_pairs, num_discordant_pairs, _, _ = concordance_index_censored(np.invert(all_censored),
                                                                                               all_target_time,
                                                                                               all_outputs_for_c_index
                                                                                               )

        # Compute AUC:
        # When computing AUC we need binary data. We don't care if a patient is censored or not as long as it has
        # A binary target. So, sorting out all the censored patients is not necessary.
        # For computing AUC we just need to get rid of all the non targeted sample
        # Sorting out all the non valid binary data:
        all_valid_indices = np.where(np.array(all_target_binary) != -1)[0]
        relevant_binary_targets = np.array(all_target_binary)[all_valid_indices]
        all_outputs = -np.array(all_outputs) if (args.target == 'Time' and args.loss == 'Cox') else all_outputs
        relevant_outputs = np.array(all_outputs)[all_valid_indices]

        fpr_train, tpr_train, _ = roc_curve(relevant_binary_targets, relevant_outputs)
        roc_auc_train = auc(fpr_train, tpr_train)

        all_writer.add_scalar('Train/Loss Per MiniBatch (Per Epoch)', train_loss / len(data_loader), e)
        all_writer.add_scalar('Train/C-index Per Epoch', c_index, e)
        all_writer.add_scalar('Train/AUC Per Epoch', roc_auc_train, e)
        all_writer.add_scalar('Train/d_Loss/d_outputs Per Epoch', dLoss__d_outputs, e)

        if e % 20 == 0:
            print('Loss: {}, C-index: {}, AUC: {}'.format(train_loss, c_index, roc_auc_train))
            # Run validation:
            test(e, test_loader)
            #test(e, test_loader_slide)
            #test(e, test_loader_patient)

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
    # is_per_patient_dataset = True if test_data_loader.dataset.is_per_patient or test_data_loader.dataset.is_per_slide else False

    model.eval()
    model.to(DEVICE)

    all_targets_time, all_targets_binary, all_outputs, all_censored = [], [], [], []
    all_patient_ids = []
    test_loss = 0

    with torch.no_grad():
        for batch_idx, minibatch in enumerate(test_data_loader):
            data = minibatch['Features']
            if len(data.shape) == 3:
                location_to_squeeze = np.where(np.array(data.shape) == 1)[0][0]
                data = data.squeeze(location_to_squeeze)

            target_time = minibatch['Time Target']
            target_binary = minibatch['Binary Target']
            censored = minibatch['Censored']

            all_targets_time.extend(target_time.numpy())
            all_targets_binary.extend(target_binary.numpy())
            all_censored.extend(censored.numpy())

            data = data.to(DEVICE)
            target_binary = target_binary.to(DEVICE)

            outputs = model(data)

            if args.target == 'Time':
                '''if is_per_patient_dataset:
                    outputs = outputs.mean()
                    all_outputs.append(outputs.item())

                else:'''
                all_outputs.extend(outputs.detach().cpu().numpy()[:, 0])
                loss = criterion(outputs, target_time, censored)

            elif args.target == 'Binary':
                outputs_after_sftmx = torch.nn.functional.softmax(outputs, dim=1)

                '''if is_per_patient_dataset:
                    outputs = outputs_after_sftmx.mean(0)  # Computing mean output for all tiles
                    all_outputs.append(outputs[1].detach().cpu().numpy())

                    if target_binary == -1:  # If the binary target for this slide/patient is -1 we cannot use it for loss computation
                        continue

                    # Finishing the computation for CrossEntropy we need to log the softmaxed outputs and then do NLLLoss
                    loss = nn.functional.nll_loss(torch.log(outputs).reshape(1, 2), torch.LongTensor([target_binary]))

                else:'''
                valid_indices = np.where(target_binary != -1)[0]
                loss = criterion(outputs[valid_indices], target_binary[valid_indices])

                all_outputs.extend(outputs_after_sftmx[:, 1].detach().cpu().numpy())

            if args.target == 'Binary':  # or not is_per_patient_dataset:
                test_loss += loss.item()

        '''if args.target == 'Time' and is_per_patient_dataset:
            test_loss = criterion(torch.tensor(all_outputs).reshape(len(all_outputs), 1), torch.tensor(all_targets_time), torch.tensor(all_censored))'''

        # Compute C index:
        all_outputs_for_c_index = -np.array(all_outputs) if flip_outputs else all_outputs
        c_index, num_concordant_pairs, num_discordant_pairs, _, _ = concordance_index_censored(np.invert(all_censored),
                                                                                               all_targets_time,
                                                                                               all_outputs_for_c_index
                                                                                               )

        # Compute AUC:

        all_valid_indices = np.where(np.array(all_targets_binary) != -1)[0]
        relevant_binary_targets = np.array(all_targets_binary)[all_valid_indices]

        # When using Cox model the scores represent Risk so we should invert them for computing AUC
        all_outputs = -np.array(all_outputs) if (args.target == 'Time' and args.loss == 'Cox') else all_outputs
        relevant_outputs = np.array(all_outputs)[all_valid_indices]

        fpr_test, tpr_test, _ = roc_curve(relevant_binary_targets, relevant_outputs)
        roc_auc_test = auc(fpr_test, tpr_test)

        if is_per_patient_dataset:
            string_end = 'Per Slide/' if test_data_loader.dataset.is_per_slide else 'Per Patient/'
            string_init = 'Test ' + string_end

        else:
            string_init = 'Test/Per Tile/'

        item_in_dset = len(test_data_loader.dataset)
        num_MB = len(test_data_loader)

        print('Validation set Performance. C-index: {}, AUC: {}, Loss(per {} num of MiniBatch): {}'.format(c_index, roc_auc_test, item_in_dset, test_loss / num_MB))

        all_writer.add_scalar(string_init + 'Loss Per MiniBatch (Per Epoch)', test_loss / num_MB, current_epoch)
        all_writer.add_scalar(string_init + 'C-index Per Epoch', c_index, current_epoch)
        all_writer.add_scalar(string_init + 'AUC Per Epoch', roc_auc_test, current_epoch)

    model.train()

########################################################################################################################
########################################################################################################################

if __name__ == '__main__':
    # Model definition:
    if args.target == 'Time':
        model = nn.Linear(512, 1)
        if args.loss == 'Cox':
            criterion = Cox_loss
            flip_outputs = False
            main_dir = 'Test_Run/Features/Time_Cox'
        elif args.loss == 'L2':
            criterion = L2_Loss
            flip_outputs = True
            main_dir = 'Test_Run/Features/Time_L2'
        else:
            Exception('No valid loss function defined')

    elif args.target == 'Binary':
        model = nn.Linear(512, 2)
        criterion = nn.CrossEntropyLoss()
        flip_outputs = True
        main_dir = 'Test_Run/Features/Binary'

    main_dir = main_dir + '_Step_' + str(args.lr) + '_Eps_' + str(args.eps) + '_WeightDecay_' + str(args.weight_decay)

    # Continue train from previous epoch:
    if args.from_epoch != 0:
        # Load model:
        raise Exception('Need to implement...')

        print('Loading pre-saved model...')
        model_data_loaded = torch.load(os.path.join(model_dir,
                                                    'model_data_Epoch_' + str(args.from_epoch) + '.pt'),
                                       map_location='cpu')

        from_epoch = args.from_epoch + 1

    else:
        from_epoch = 0

    check_parameters = False
    check_optimization = False
    check_Ran_model = False
    check_own_model = False
    train_from_Ran_parameters = False

    if check_parameters or check_optimization:
        model.bias.data = torch.ones(1) * bias
        for digit_num in range(model.weight.data.size(1)):
            model.weight.data[0][digit_num] = torch.ones(1) * parameters[digit_num]

        if check_parameters:
            model.eval()
        elif check_optimization:
            optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=5e-5)
            model.train()

    elif check_Ran_model:
        model_location = r'/Users/wasserman/Developer/WSI_MIL/Data from gipdeep/runs/Ran_models/survival/Ran_Exp_20094-survival-TestFold_1/model_data_Epoch_1000.pt'
        basic_model_data = torch.load(model_location, map_location='cpu')['model_state_dict']
        basic_model = PreActResNet50_Ron()
        basic_model.load_state_dict(basic_model_data)
        last_linear_layer_data = copy.deepcopy(basic_model.linear.state_dict())
        model.load_state_dict(last_linear_layer_data)
        model.eval()

    elif check_own_model:
        model_location = r'/Users/wasserman/Developer/WSI_MIL/Test_Run/Features/Binary_Step_1e-05/Model_CheckPoints/model_data_Epoch_500.pt'
        model_data = torch.load(model_location, map_location='cpu')['model_state_dict']
        model.load_state_dict(model_data)
        model.eval()

    else:
        if train_from_Ran_parameters:
            print('\x1b[0;30;42m' + 'Training from Ran parameters....' + '\x1b[0m')
            model_location = r'/Users/wasserman/Developer/WSI_MIL/Data from gipdeep/runs/Ran_models/survival/Ran_Exp_20094-survival-TestFold_1/model_data_Epoch_1000.pt'
            basic_model_data = torch.load(model_location, map_location='cpu')['model_state_dict']
            basic_model = PreActResNet50_Ron()
            basic_model.load_state_dict(basic_model_data)
            last_linear_layer_data = copy.deepcopy(basic_model.linear.state_dict())
            model.load_state_dict(last_linear_layer_data)
            model.train()

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=args.eps)
        #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [300], gamma=0.1)
        #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50)

    DEVICE = utils.device_gpu_cpu()
    num_workers = utils.get_cpu()

    # Initializing Datasets
    test_set = Features_to_Survival(is_train=False)
    #test_set_slide = Features_to_Survival(is_train=False, is_per_slide=True)
    #test_set_patient = Features_to_Survival(is_train=False, is_per_patient=True)

    if args.censored_ratio < 0:  # We're using the data as is and not manipulating the amount of censored/not censored samples in each minibatch
        train_set = Features_to_Survival(is_train=True)

    elif args.censored_ratio == 0:  # All samples are NOT censored
        train_set = Features_to_Survival(is_train=True, is_all_not_censored=True)

    elif args.censored_ratio > 0 and args.censored_ratio < 1:
        train_set_censored = Features_to_Survival(is_train=True, is_all_censored=True)
        train_set_not_censored = Features_to_Survival(is_train=True, is_all_not_censored=True)

    # Initializing DataLoaders:
    test_loader = DataLoader(test_set, batch_size=args.mini_batch_size, shuffle=False, num_workers=num_workers)
    #test_loader_slide = DataLoader(test_set_slide, batch_size=1, shuffle=False, num_workers=num_workers)
    #test_loader_patient = DataLoader(test_set_patient, batch_size=1, shuffle=False, num_workers=num_workers)

    if args.censored_ratio <= 0:
        train_loader = {'Censored': DataLoader(train_set, batch_size=args.mini_batch_size, shuffle=True),
                        'Not Censored': cycle([0])
                        }

    elif args.censored_ratio > 0 and args.censored_ratio < 1:
        train_loader_censored = DataLoader(train_set_censored, batch_size=args.censored, shuffle=True, drop_last=True, num_workers=num_workers)
        train_loader_not_censored = DataLoader(train_set_not_censored, batch_size=args.not_censored, shuffle=True, drop_last=True, num_workers=num_workers)
        if len(train_loader_censored) < len(train_loader_not_censored):
            train_loader_censored = cycle(train_loader_censored)
        elif len(train_loader_censored) > len(train_loader_not_censored):
            train_loader_not_censored = cycle(train_loader_not_censored)

        train_loader = {'Censored': train_loader_censored,
                        'Not Censored': train_loader_not_censored
                        }

    if not os.path.isdir(os.path.join(main_dir, 'Model_CheckPoints')):
        Path(os.path.join(main_dir, 'Model_CheckPoints')).mkdir(parents=True)

    all_writer = SummaryWriter(main_dir)

    if check_parameters:
        test(current_epoch=1, test_data_loader=train_loader)

    elif check_Ran_model or check_own_model:
        # Initializing train set for tile, slide, patient
        train_set_slide = Features_to_Survival(is_train=True, is_per_slide=True)
        train_set_patient = Features_to_Survival(is_train=True, is_per_patient=True)

        train_loader = DataLoader(train_set, batch_size=50, shuffle=False)
        train_loader_slide = DataLoader(train_set_slide, batch_size=1, shuffle=False)
        train_loader_patient = DataLoader(train_set_patient, batch_size=1, shuffle=False)

        if check_own_model:
            print('Checking ' + '\x1b[0;30;42m' + 'own' + '\x1b[0m' + 'model {} parameters'.format(model_location))
        elif check_Ran_model:
            print('Checking ' + '\x1b[0;30;42m' + 'Ran' + '\x1b[0m' + 'models parameters')

        print('Per Tile Performance over ' + '\x1b[0;30;44m' + 'Train' + '\x1b[0m' + ' Set:')
        test(current_epoch=1, test_data_loader=train_loader)
        print('Per Slide Performance over ' + '\x1b[0;30;44m' + 'Train' + '\x1b[0m' + ' Set:')
        test(current_epoch=1, test_data_loader=train_loader_slide)
        print('Per Patient Performance over ' + '\x1b[0;30;44m' + 'Train' + '\x1b[0m' + ' Set:')
        test(current_epoch=1, test_data_loader=train_loader_patient)

        print('Per Tile Performance over ' + '\x1b[0;30;41m' + 'Test' + '\x1b[0m' + ' Set:')
        test(current_epoch=1, test_data_loader=test_loader)
        print('Per Slide Performance over ' + '\x1b[0;30;41m' + 'Test' + '\x1b[0m' + ' Set:')
        test(current_epoch=1, test_data_loader=test_loader_slide)
        print('Per Patient Performance over ' + '\x1b[0;30;41m' + 'Test' + '\x1b[0m' + ' Set:')
        test(current_epoch=1, test_data_loader=test_loader_patient)

    else:
        train(from_epoch=from_epoch, epochs=args.epochs, data_loader=train_loader)

    print('Done')

