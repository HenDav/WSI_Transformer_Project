from sksurv.metrics import concordance_index_censored

import Omer_files_suspected_as_unnecessary.omer_utils
from datasets import WSI_REGdataset_Survival
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
from Survival.Cox_Loss import Cox_loss, L2_Loss
import torch.optim as optim
import utils
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import argparse
from sklearn.metrics import roc_curve, auc
import os
from pathlib import Path
from Nets.PreActResNets import PreActResNet50_Ron
import sys
import time
import psutil
from datetime import datetime
from itertools import cycle
import pandas as pd


utils.send_run_data_via_mail()

parser = argparse.ArgumentParser(description='')
parser.add_argument('-ex', '--experiment', type=int, default=0, help='Continue train of this experiment')
parser.add_argument('-fe', '--from_epoch', type=int, default=0, help='Continue train from epoch')
parser.add_argument('-tf', '--test_fold', default=1, type=int, help='fold to be as TEST FOLD')
parser.add_argument('--transform_type', default='rvf', type=str, help='none / flip / wcfrs (weak color+flip+rotate+scale)')
parser.add_argument('-e', '--epochs', default=500, type=int, help='Epochs to run')
parser.add_argument('-tar', '--target', type=str, default='Time', help='Binary / Time')
parser.add_argument('-l', '--loss', type=str, default='Cox', help='Cox / L2')
parser.add_argument('-mb', '--mini_batch_size', type=int, default=18, help='Mini batch size')
parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
parser.add_argument('-wd', '--weight_decay', default=5e-5, type=float, help='L2 penalty')
parser.add_argument('-time', dest='time', action='store_true', help='save train timing data ?')
parser.add_argument('-cr', '--censored_ratio', type=float, default=0.5, help='ratio of censored samples in each minibatch')
parser.add_argument('-se', '--save_every', default=20, type=int, help='duration to save models (in epochs)')
parser.add_argument('-sxl', '--save_xl_every', default=5, type=int, help='duration to save excel file (in epochs)')

parser.add_argument('--debug', action='store_true', help='debug ?')
args = parser.parse_args()

if sys.platform == 'darwin':
    args.debug = True

args.loss = args.loss if args.target == 'Time' else ''  # This modification is needed only to fix the data written in the code files (run parameters)

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
    if args.time:
        times = {}

    num_minibatch = len(data_loader['Not Censored']) if type(data_loader['Censored']) == cycle else len(data_loader['Censored'])

    for e in range(from_epoch, from_epoch + epochs):
        model.train()
        model.to(DEVICE)
        print('Epoch {}:'.format(e))
        all_target_time, all_target_binary, all_outputs, all_censored = [], [], [], []
        targets_binary_xl, targets_time_xl, scores_xl, slides_xl, censored_xl = [], [], [], [], []  # FIXME: These are debug variables
        train_loss = 0
        dLoss__d_outputs = 0
        nan_count = 0

        process = psutil.Process(os.getpid())
        print('RAM usage:', np.round(process.memory_info().rss / 1e9), 'GB, time: ', datetime.now())

        #for batch_idx, minibatch in enumerate(tqdm(data_loader)):
        for batch_idx, minibatch in enumerate(tqdm(zip(data_loader['Censored'], data_loader['Not Censored']))):
            time_stamp = batch_idx + e * num_minibatch
            if args.time:
                time_start = time.time()

            minibatch = Omer_files_suspected_as_unnecessary.omer_utils.concatenate_minibatch(minibatch, is_shuffle=True)

            data = minibatch['Data']
            target_time = minibatch['Time Target']
            target_binary = minibatch['Binary Target']
            censored = minibatch['Censored']

            # TODO: DEBUG - save data to be saved later on in xl file
            slide_names = minibatch['File Names']
            targets_binary_xl.extend(target_binary[:, 0].detach().cpu().numpy())
            targets_time_xl.extend(target_time.detach().cpu().numpy())
            slides_xl.extend(slide_names)
            censored_xl.extend(censored.detach().cpu().numpy())

            censored_ratio = len(np.where(censored == True)[0]) / len(censored)
            #if (args.censored_ratio > 0 and censored_ratio != args.censored_ratio) or len(censored) != args.mini_batch_size:
            if args.censored_ratio > 0 and (censored_ratio != args.censored_ratio or len(censored) != args.mini_batch_size):
                print('Censored Ratio is {}, MB Size is {}'.format(censored_ratio, len(censored)))
                raise Exception('Problem occured in batch size or censored ratio')

            all_writer.add_scalar('Train/Censored Ratio per Minibatch', censored_ratio, time_stamp)

            if args.time:
                times['Tile Extraction'] = minibatch['Time dict']['Average time to extract a tile'].detach().cpu().numpy().mean()
                times['Tile Augmentation'] = minibatch['Time dict']['Augmentation time'].detach().cpu().numpy().mean()
                times['Total Tile'] = minibatch['Time dict']['Total time'].detach().cpu().numpy().mean()

            all_target_binary.extend(target_binary.numpy())
            all_target_time.extend(target_time.numpy())
            all_censored.extend(censored.numpy())

            data = data.to(DEVICE)

            optimizer.zero_grad()

            if args.time:
                times['Data manipulation after tile extraction'] = time.time() - time_start

            if args.time:
                time_start_fwd_pass = time.time()

            outputs, _ = model(data)
            outputs.retain_grad()

            # TODO: DEBUG - save data to be save in xl file
            scores_xl.extend(outputs[:, 0].detach().cpu().numpy())

            if args.time:
                times['Forward Pass'] = time.time() - time_start_fwd_pass
                time_start_loss_calc = time.time()

            if args.target == 'Time':
                all_outputs.extend(outputs.detach().cpu().numpy()[:, 0])
                loss = criterion(outputs, target_time.to(DEVICE), censored)
                if np.isnan(loss.item()):
                    nan_count += 1
                    #print('Got Nan in Loss computation. Check Censor ratio')
                    #print('Epoch No. {}, Censor Ratio is {}, MB Size is {}'.format(e, censored_ratio, len(censored)))


            elif args.target == 'Binary':
                outputs_after_sftmx = torch.nn.functional.softmax(outputs, dim=1)
                all_outputs.extend(outputs_after_sftmx[:, 1].detach().cpu().numpy())

                # For the Binary Target case, we can compute loss only for samples with valid binary target (not '-1')
                valid_indices = np.where(target_binary != -1)[0]
                target_binary = target_binary.to(DEVICE)
                loss = criterion(outputs[valid_indices], target_binary[valid_indices].squeeze(1))

            train_loss += loss.item()
            all_writer.add_scalar('Train/Loss per Minibatch', loss.item(), time_stamp)

            if args.time:
                times['Loss Calculation'] = time.time() - time_start_loss_calc
                time_start_backprop = time.time()

            loss.backward()
            dLoss__d_outputs += np.sum(np.abs(outputs.grad.detach().cpu().numpy()))
            optimizer.step()

            if args.time:
                times['Backprop'] = time.time() - time_start_backprop

                all_writer.add_scalar('Time/Avg to Extract Tile [Sec]', times['Tile Extraction'], time_stamp)
                all_writer.add_scalar('Time/Augmentation [Sec]', times['Tile Augmentation'], time_stamp)
                all_writer.add_scalar('Time/Total To Collect Data [Sec]', times['Total Tile'], time_stamp)
                all_writer.add_scalar('Time/Data Manipulations [Sec]', times['Data manipulation after tile extraction'], time_stamp)
                all_writer.add_scalar('Time/Loss Calc [Sec]', times['Loss Calculation'], time_stamp)
                all_writer.add_scalar('Time/Forward Pass [Sec]', times['Forward Pass'], time_stamp)
                all_writer.add_scalar('Time/Back Propagation [Sec]', times['Backprop'], time_stamp)

            if DEVICE.type == 'cuda':
                res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
                all_writer.add_scalar('GPU/gpu', res.gpu, time_stamp)
                all_writer.add_scalar('GPU/gpu-mem', res.memory, time_stamp)

            if args.time:
                time_full_minibatch = time.time() - time_start
                all_writer.add_scalar('Time/Full Minibatch [Sec]', time_full_minibatch, time_stamp)

        if nan_count != 0:
            print('Epoch {}, Nan count {} / {}'.format(e, nan_count, num_minibatch))
        if args.time:
            time_start_performance_calc = time.time()

        # Compute C index:
        # PAY ATTENTION: the function 'concordance_index_censored' takes censored = True as not censored (This is why we should invert 'all_censored' )
        # and takes the outputs as a risk NOT as survival time !!!!!!!!!!
        all_outputs_for_c_index = -np.array(all_outputs) if flip_outputs else all_outputs
        c_index, num_concordant_pairs, num_discordant_pairs, _, _ = concordance_index_censored(np.invert(all_censored),
                                                                                               all_target_time,
                                                                                               all_outputs_for_c_index
                                                                                               )

        # Compute AUC:
        # We computer AUC w.r.t the binary targets so we need to sort out samples with binary target of -1
        all_valid_indices = np.where(np.array(all_target_binary) != -1)[0]
        relevant_binary_targets = np.array(all_target_binary)[all_valid_indices]

        # When using Cox model the scores represent Risk so we should invert them for computing AUC
        all_outputs = -np.array(all_outputs) if (args.target == 'Time' and args.loss == 'Cox') else np.array(all_outputs)
        relevant_outputs = all_outputs[all_valid_indices]

        fpr_train, tpr_train, _ = roc_curve(relevant_binary_targets, relevant_outputs)
        auc_train = auc(fpr_train, tpr_train)

        #print('(Train) C-index: {}, AUC: {}'.format(c_index, auc_train))
        #print(len(scores_xl), len(slides_xl), len(targets_binary_xl), len(targets_time_xl), len(censored_xl))

        if e % args.save_xl_every == 0:
            xl_dict = {'Scores': scores_xl,
                       'Slide Names': slides_xl,
                       'Targets Binary': targets_binary_xl,
                       'Targets Time': targets_time_xl,
                       'Censored': censored_xl}

            xl_DF = pd.DataFrame(xl_dict)
            xl_DF.to_excel(os.path.join(args.output_dir, 'debug_outputs_epoch_' + str(e) + '.xlsx'))


        if args.time:
            all_writer.add_scalar('Time/Performance calc time', time.time() - time_start_performance_calc, e)

        # The loss is normalized per sample for each minibatch. We sum all loss for each minibatch so if we want to get
        # the mean loss per sample we need to divide train_loss by the number of minibatches

        all_writer.add_scalar('Train/Loss Per Sample (Per Epoch)', train_loss / num_minibatch, e)
        all_writer.add_scalar('Train/C-index Per Epoch', c_index, e)
        all_writer.add_scalar('Train/AUC Per Epoch', auc_train, e)
        all_writer.add_scalar('Train/d_Loss/d_outputs Per Epoch', dLoss__d_outputs, e)

        print('(Train) Loss: {}, C-index: {}, AUC: {}'.format(train_loss, c_index, auc_train))
        if e % args.save_every == 0:
            # Run validation:
            test(e, test_loader)

            utils.run_data(experiment=experiment, epoch=e)
            # Save model to file:
            try:
                model_state_dict = model.module.state_dict()
            except AttributeError:
                model_state_dict = model.state_dict()

            torch.save({'epoch': e,
                        'model_state_dict': model_state_dict
                        },
                       os.path.join(args.output_dir, 'Model_CheckPoints', 'model_data_Epoch_' + str(e) + '.pt'))

        if args.debug:
            test(e, train_loader_debug, debug=True)
            test(e, train_loader, debug=True)


    all_writer.close()


def test(current_epoch, test_data_loader, debug: bool = False):
    model.eval()
    model.to(DEVICE)

    all_targets_time, all_targets_binary, all_outputs, all_censored = [], [], [], []
    test_loss = 0

    with torch.no_grad():
        for batch_idx, minibatch in enumerate(test_data_loader):
            if type(test_data_loader) != torch.utils.data.dataloader.DataLoader:
                minibatch = Omer_files_suspected_as_unnecessary.omer_utils.concatenate_minibatch(minibatch, is_shuffle=False)

            data = minibatch['Data']
            target_time = minibatch['Time Target']
            target_binary = minibatch['Binary Target']
            censored = minibatch['Censored']

            all_targets_time.extend(target_time.numpy())
            all_targets_binary.extend(target_binary.numpy())
            all_censored.extend(censored.numpy())

            data = data.to(DEVICE)
            target_time = target_time.to(DEVICE)

            outputs, _ = model(data)

            if args.target == 'Time':
                loss = criterion(outputs, target_time, censored)
                all_outputs.extend(outputs.detach().cpu().numpy()[:, 0])

            elif args.target == 'Binary':
                # For the Binary Target case, we can compute loss only for samples with valid binary target (not '-1')
                valid_indices = np.where(target_binary != -1)[0]
                target_binary = target_binary.to(DEVICE)
                loss = criterion(outputs[valid_indices], target_binary[valid_indices].squeeze(1))

                outputs_after_sftmx = torch.nn.functional.softmax(outputs, dim=1)
                all_outputs.extend(outputs_after_sftmx[:, 1].detach().cpu().numpy())

            test_loss += loss.item()

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
        all_outputs = -np.array(all_outputs) if (args.target == 'Time' and args.loss == 'Cox') else np.array(all_outputs)
        relevant_outputs = all_outputs[all_valid_indices]

        fpr_test, tpr_test, _ = roc_curve(relevant_binary_targets, relevant_outputs)
        auc_test = auc(fpr_test, tpr_test)

        if debug:
            print('Debug set Performance. C-index: {}, AUC: {}'.format(c_index, auc_test))
        else:
            print('Validation set Performance. C-index: {}, AUC: {}'.format(c_index, auc_test))
        if not debug:
            all_writer.add_scalar('Test/Loss Per Sample (Per Epoch)', test_loss / len(test_data_loader), current_epoch)
            all_writer.add_scalar('Test/C-index Per Epoch', c_index, current_epoch)
            all_writer.add_scalar('Test/AUC Per Epoch', auc_test, current_epoch)

    model.train()


def initialize_trainset(censor_ratio):
    if censor_ratio < 0:  # We're using the data as is and not manipulating the amount of censored/not censored samples in each minibatch
        train_dset = WSI_REGdataset_Survival(train=True,
                                             DataSet='ABCTB',
                                             tile_size=TILE_SIZE,
                                             target_kind='survival',
                                             test_fold=args.test_fold,
                                             transform_type=args.transform_type
                                             )
        num_train_samples = train_dset.real_length

        loader_censored = DataLoader(train_dset,
                                     batch_size=args.mini_batch_size,
                                     shuffle=True,
                                     num_workers=num_workers,
                                     pin_memory=True)
        loader_not_censored = cycle([0])

        train_loader = {'Censored': loader_censored,
                        'Not Censored': loader_not_censored
                        }

    elif censor_ratio == 0:  # All samples are NOT censored
        train_dset = WSI_REGdataset_Survival(train=True,
                                             DataSet='ABCTB',
                                             tile_size=TILE_SIZE,
                                             target_kind='survival',
                                             test_fold=args.test_fold,
                                             transform_type=args.transform_type,
                                             is_all_not_censored=True
                                             )
        num_train_samples = train_dset.real_length

        loader_censored = DataLoader(train_dset,
                                     batch_size=args.mini_batch_size,
                                     shuffle=True,
                                     num_workers=num_workers,
                                     pin_memory=True)
        loader_not_censored = cycle([0])

        train_loader = {'Censored': loader_censored,
                        'Not Censored': loader_not_censored
                        }

    elif censor_ratio > 0 and censor_ratio < 1:
        train_dset_censored = WSI_REGdataset_Survival(train=True,
                                                      DataSet='ABCTB',
                                                      tile_size=TILE_SIZE,
                                                      target_kind='survival',
                                                      test_fold=args.test_fold,
                                                      transform_type=args.transform_type,
                                                      is_all_censored=True
                                                      )
        train_dset_not_censored = WSI_REGdataset_Survival(train=True,
                                                          DataSet='ABCTB',
                                                          tile_size=TILE_SIZE,
                                                          target_kind='survival',
                                                          test_fold=args.test_fold,
                                                          transform_type=args.transform_type,
                                                          is_all_not_censored=True
                                                          )
        num_train_samples = train_dset_censored.real_length + train_dset_not_censored.real_length

        loader_censored = DataLoader(train_dset_censored,
                                     batch_size=args.censored,
                                     shuffle=True,
                                     num_workers=num_workers,
                                     drop_last=True,
                                     pin_memory=True)
        loader_not_censored = DataLoader(train_dset_not_censored,
                                         batch_size=args.not_censored,
                                         shuffle=True,
                                         num_workers=num_workers,
                                         drop_last=True,
                                         pin_memory=True)

        if len(loader_censored) < len(loader_not_censored):
            loader_censored = cycle(loader_censored)

        elif len(loader_censored) > len(loader_not_censored):
            loader_not_censored = cycle(loader_not_censored)

        train_loader = {'Censored': loader_censored,
                        'Not Censored': loader_not_censored
                        }

########################################################################################################################
########################################################################################################################

if __name__ == '__main__':

    DEVICE = utils.device_gpu_cpu()
    num_workers = utils.get_cpu()

    if DEVICE.type == 'cuda':
        import nvidia_smi
        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

    TILE_SIZE = 128 if sys.platform == 'darwin' else 256

    # Getting the receptor name:
    receptor = 'Survival_' + args.target
    if args.target == 'Time':
        receptor += '_' + args.loss

    if args.experiment == 0:
        run_data_results = utils.run_data(test_fold=args.test_fold,
                                          transform_type=args.transform_type,
                                          tile_size=TILE_SIZE,
                                          tiles_per_bag=1,
                                          DataSet_name='ABCTB',
                                          Receptor=receptor,
                                          num_bags=args.mini_batch_size,
                                          learning_rate=args.lr,
                                          censored_ratio=args.censored_ratio)

        args.output_dir, experiment = run_data_results['Location'], run_data_results['Experiment']

    else:
        run_data_output = utils.run_data(experiment=args.experiment)
        args.output_dir, args.test_fold, args.transform_type, TILE_SIZE, tiles_per_bag, \
        args.batch_size, args.dx, args.dataset, args.target, args.model, args.mag = \
            run_data_output['Location'], run_data_output['Test Fold'], run_data_output['Transformations'], \
            run_data_output['Tile Size'], run_data_output['Tiles Per Bag'], run_data_output['Num Bags'],\
            run_data_output['DX'], run_data_output['Dataset Name'], run_data_output['Receptor'],\
            run_data_output['Model Name'], run_data_output['Desired Slide Magnification']

        if args.target == 'Survival_Binary':
            args.target = 'Binary'
        elif args.target in ['Features_Survival_Time_Cox', 'Survival_Time_Cox', 'Survival_Time_L2']:
            args.target = 'Time'
            if args.target in ['Features_Survival_Time_Cox', 'Survival_Time_Cox']:
                args.loss = 'Cox'
            elif args.target == 'Survival_Time_L2':
                args.loss = 'L2'

        print('args.dataset:', args.dataset)
        print('args.target:', args.target)
        print('args.args.batch_size:', args.batch_size)
        print('args.output_dir:', args.output_dir)
        print('args.test_fold:', args.test_fold)
        print('args.transform_type:', args.transform_type)
        print('args.dx:', args.dx)
        if args.target == 'Time':
            print('args.loss:', args.loss)

        experiment = args.experiment

    # Model definition:
    model = PreActResNet50_Ron()

    if args.target == 'Time':
        model.linear = nn.Linear(model.linear.in_features, 1)
        if args.loss == 'Cox':
            criterion = Cox_loss
            flip_outputs = False

        elif args.loss == 'L2':
            criterion = L2_Loss
            flip_outputs = True

        else:
            Exception('No valid loss function defined')

    elif args.target == 'Binary':
        criterion = nn.CrossEntropyLoss()
        flip_outputs = True

    # Continue train from previous epoch:
    if args.from_epoch != 0:
        print('Loading pre-saved model...')
        model_data_loaded = torch.load(os.path.join(args.output_dir, 'Model_CheckPoints',
                                                    'model_data_Epoch_' + str(args.from_epoch) + '.pt'),
                                       map_location='cpu')

        from_epoch = args.from_epoch + 1
        model.load_state_dict(model_data_loaded['model_state_dict'])

    else:
        from_epoch = 0

    # Initializing Datasets and DataLoaders:
    censored_train_loader, not_censored_train_loader = initialize_trainset(censor_ratio=0.5)


    test_dset = WSI_REGdataset_Survival(train=False,
                                        DataSet='ABCTB',
                                        tile_size=TILE_SIZE,
                                        target_kind='survival',
                                        test_fold=args.test_fold,
                                        transform_type='none'
                                        )

    test_loader = DataLoader(test_dset, batch_size=args.mini_batch_size * 2, shuffle=False, num_workers=num_workers, pin_memory=True)

    if args.debug:  # create another train set
        train_dset_debug = WSI_REGdataset_Survival(train=True,
                                                   DataSet='ABCTB',
                                                   tile_size=TILE_SIZE,
                                                   target_kind='survival',
                                                   test_fold=args.test_fold,
                                                   transform_type=args.transform_type
                                                   )

        train_loader_debug = DataLoader(train_dset_debug, batch_size=args.mini_batch_size * 2, shuffle=False, num_workers=num_workers, pin_memory=True)


    # Save transformation data to 'run_data.xlsx'
    train_dset_to_check = train_dset if 'train_dset' in locals() else train_dset_censored
    transformation_string = ', '.join([str(train_dset_to_check.transform.transforms[i]) for i in range(len(train_dset_to_check.transform.transforms))])
    utils.run_data(experiment=experiment, transformation_string=transformation_string)

    # Save model data and data-set size to run_data.xlsx file.  # FIXME: Change this if you want to train a trained model
    utils.run_data(experiment=experiment, model=model.model_name)
    utils.run_data(experiment=experiment, DataSet_size=(num_train_samples, test_dset.real_length))
    utils.run_data(experiment=experiment, DataSet_Slide_magnification=train_dset_to_check.desired_magnification)

    # Saving code files, args and main file name (this file) to Code directory within the run files.
    utils.save_code_files(args, train_dset_to_check)

    all_writer = SummaryWriter(os.path.join(args.output_dir, 'writer'))

    if not os.path.isdir(os.path.join(args.output_dir, 'Model_CheckPoints')):
        Path(os.path.join(args.output_dir, 'Model_CheckPoints')).mkdir(parents=True)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train(from_epoch=from_epoch, epochs=args.epochs, data_loader=train_loader)

    print('Done')

