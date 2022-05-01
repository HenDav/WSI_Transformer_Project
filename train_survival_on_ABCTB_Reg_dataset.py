from sksurv.metrics import concordance_index_censored
from datasets import WSI_REGdataset_Survival_CR
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
from Cox_Loss import Cox_loss, L2_Loss, Combined_loss
import torch.optim as optim
import utils
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import argparse
from sklearn.metrics import roc_curve, auc
import os
from pathlib import Path
from PreActResNets import PreActResNet50_Ron
import sys
import time
import psutil
from datetime import datetime
from itertools import cycle
from random import shuffle
import pandas as pd
from torch import linalg as LA

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
parser.add_argument('-cr', '--censored_ratio', type=float, default=0, help='Mean ratio of censored samples in each minibatch')
parser.add_argument('-se', '--save_every', default=20, type=int, help='duration to save models (in epochs)')
parser.add_argument('-sxl', '--save_xl_every', default=1e5, type=int, help='duration to save excel file (in epochs)')
parser.add_argument('-lw', '--loss_weights', nargs="+", default=[1/3.2, 1/1382, 1], help='loss weights - [Cox, L2, Binary]')
parser.add_argument('-loss_comb', dest='loss_combination', action='store_true', help='Combine all 3 loss functions ?')
parser.add_argument('-DEBUG_pg', dest='print_gradients', action='store_true', help='')
args = parser.parse_args()

if args.print_gradients:
    args.loss_combination = True
    args.test_fold = 1
    args.epochs = 10
    args.mini_batch_size = 18
    args.censored_ratio = 0.5
    args.loss_weights = [1, 1, 1]

# Convert items in args.loss_weights to floats
temp = []
for value in args.loss_weights:
    if type(value) in [float, int]:
        temp.append(float(value))
    elif type(value) == str:
        temp.append(eval(value))

args.loss_weights_as_str = args.loss_weights
args.loss_weights = temp

# The following modification is needed only to fix the data written in the code files (run parameters)
args.loss = args.loss if args.target == 'Time' else ''

if args.censored_ratio == 1:
    raise Exception('The value of censored_ratio CANNOT be 1 since all samples will be censored and that is unacceptable')
elif args.censored_ratio > 1:
    raise Exception('The value of censored_ratio CANNOT be grater than 1')
elif args.censored_ratio < 0:
    print('Dataset will be initiated as is (without any manipulations of censored/ not censored number of samples in the minibatch)')


def train(from_epoch: int = 0, epochs: int = 2, data_loader=None):
    if args.time:
        times = {}

    size_minibatch = len(data_loader)  #len(data_loader['Not Censored']) if type(data_loader['Censored']) == cycle else len(data_loader['Censored'])


    for e in range(from_epoch, from_epoch + epochs):
        model.train()
        model.to(DEVICE)
        print('Epoch {}:'.format(e))

        if args.loss_combination:
            all_outputs_Cox, all_outputs_L2, all_outputs_Binary = [], [], []
            train_loss, loss_cox_total, loss_L2_total, loss_cross_entropy_total = 0, 0, 0, 0
            dLoss__d_outputs_cox, dLoss__d_outputs_L2, dLoss__d_outputs_cross_entropy = 0, 0, 0
            dLoss__d_outputs_cox_times_weights, dLoss__d_outputs_L2_times_weights, dLoss__d_outputs_cross_entropy_times_weights = 0, 0, 0

        all_target_time, all_target_binary, all_censored, all_outputs = [], [], [], []
        gpu_mem, gpu = [], []

        targets_binary_xl, targets_time_xl, scores_xl, slides_xl, censored_xl = [], [], [], [], []  # FIXME: These are debug variables
        train_loss = 0
        dLoss__d_outputs = 0
        nan_count = 0
        samples_per_epoch = 0

        process = psutil.Process(os.getpid())
        print('RAM usage:', np.round(process.memory_info().rss / 1e9), 'GB, time: ', datetime.now())

        #for batch_idx, minibatch in enumerate(tqdm(zip(data_loader['Censored'], data_loader['Not Censored']))):
        # for batch_idx, minibatch in enumerate(data_loader):
        for batch_idx, minibatch in enumerate(tqdm(data_loader)):
            time_stamp = batch_idx + e * size_minibatch
            if args.time:
                time_start = time.time()

            #minibatch = utils.concatenate_minibatch(minibatch, is_shuffle=True)

            censored = minibatch['Censored']
            samples_per_epoch += len(censored)
            num_not_censored = len(np.where(censored == False)[0])
            if num_not_censored < 2:  # Skipp this minibatch if there are less than 2 "Not Censored" tiles in the minibatch
                continue

            data = minibatch['Data']
            target_time = minibatch['Time Target']
            target_binary = minibatch['Binary Target']

            # TODO: DEBUG - save data to be saved later on in xl file
            slide_names = minibatch['File Names']
            targets_binary_xl.extend(target_binary[:, 0].detach().cpu().numpy())
            targets_time_xl.extend(target_time.detach().cpu().numpy())
            slides_xl.extend(slide_names)
            censored_xl.extend(censored.detach().cpu().numpy())

            '''#if (args.censored_ratio > 0 and censored_ratio != args.censored_ratio) or len(censored) != args.mini_batch_size:
            if args.censored_ratio > 0 and (censored_ratio != args.censored_ratio or len(censored) != args.mini_batch_size):
                print('Censored Ratio is {}, MB Size is {}'.format(censored_ratio, len(censored)))
                raise Exception('Problem occured in batch size or censored ratio')'''

            censored_ratio = len(np.where(censored == True)[0]) / len(censored)
            all_writer.add_scalar('Train/Censored Ratio per Minibatch', censored_ratio, time_stamp)

            if args.time:
                times['Tile Extraction'] = minibatch['Time dict']['Average time to extract a tile'].detach().cpu().numpy().mean()
                times['Tile Augmentation'] = minibatch['Time dict']['Augmentation time'].detach().cpu().numpy().mean()
                times['Total Tile'] = minibatch['Time dict']['Total time'].detach().cpu().numpy().mean()

            all_target_binary.extend(target_binary.numpy())
            all_target_time.extend(target_time.numpy())
            all_censored.extend(censored.numpy())

            data = data.to(DEVICE)
            target_binary = target_binary.to(DEVICE)
            target_time = target_time.to(DEVICE)

            optimizer.zero_grad()

            if args.time:
                times['Data manipulation after tile extraction'] = time.time() - time_start
                time_start_fwd_pass = time.time()

            outputs, features = model(data)
            outputs.retain_grad()

            if args.print_gradients:
                features.retain_grad()

            if args.time:
                times['Forward Pass'] = time.time() - time_start_fwd_pass
                time_start_loss_calc = time.time()

            if args.loss_combination:
                all_outputs_Cox.extend(outputs.detach().cpu().numpy()[:, 0])
                all_outputs_L2.extend(outputs.detach().cpu().numpy()[:, 1])
                outputs_for_binary = torch.nn.functional.softmax(outputs[:, 2:], dim=1)
                all_outputs_Binary.extend(outputs_for_binary[:, 1].detach().cpu().numpy())

                loss, loss_cox, loss_L2, loss_cross_entropy = criterion(outputs,
                                                                        targets_time=target_time,
                                                                        targets_binary=target_binary,
                                                                        censored=censored,
                                                                        weights=args.loss_weights)
                if np.isnan(loss_cox.item()):
                    nan_count += 1
            else:
                scores_xl.extend(outputs[:, 0].detach().cpu().numpy())  # DEBUG - save data to be save in xl file

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
                    valid_indices = np.where(target_binary.detach().cpu().numpy() != -1)[0]
                    loss = criterion(outputs[valid_indices], target_binary[valid_indices].squeeze(1))

            if args.loss_combination:
                loss_cox_total += loss_cox.item()
                loss_L2_total += loss_L2.item()
                loss_cross_entropy_total += loss_cross_entropy.item()

            train_loss += loss.item()
            all_writer.add_scalar('Train/Loss per Minibatch', loss.item(), time_stamp)

            if args.time:
                times['Loss Calculation'] = time.time() - time_start_loss_calc
                time_start_backprop = time.time()

            loss.backward()


            if e == 3 and args.print_gradients:
                print('Epoch No. {}, minibatch No. {}'.format(e+1, batch_idx+1))
                save_debug_data(outputs, features)

            if args.loss_combination:
                dLoss__d_outputs_cox += np.sum(np.abs(outputs.grad.detach().cpu().numpy()[:, 0]))
                dLoss__d_outputs_L2 += np.sum(np.abs(outputs.grad.detach().cpu().numpy()[:, 1]))
                dLoss__d_outputs_cross_entropy += np.sum(np.abs(outputs.grad.detach().cpu().numpy()[:, 2]))

                # Getting the values of the linear layer and subtracting dimension 4 from dimension 3
                weights_liner_layer = torch.zeros(3, 512)
                weights_liner_layer[:2] = model.linear.weight[:2]
                weights_liner_layer[2] = model.linear.weight[2] - model.linear.weight[3]

                dLoss__d_outputs_mean = np.zeros(3)
                dLoss__d_outputs_mean[0] = torch.mean(torch.abs(outputs.grad[:, 0])).item()
                dLoss__d_outputs_mean[1] = torch.mean(torch.abs(outputs.grad[:, 1])).item()
                dLoss__d_outputs_mean[2] = torch.mean(torch.abs(outputs.grad[:, 3])).item()

                #weights_norm_liner_layer_torch = LA.vector_norm(weights_liner_layer, dim=1, ord=2)  # not working on gipdeep
                weights_norm_liner_layer = np.linalg.norm(weights_liner_layer.detach().cpu().numpy(), axis=1, ord=2)

                d_loss_times_linear_weights = weights_norm_liner_layer * dLoss__d_outputs_mean

                dLoss__d_outputs_cox_times_weights += np.sum(d_loss_times_linear_weights[0].item())
                dLoss__d_outputs_L2_times_weights += np.sum(d_loss_times_linear_weights[1].item())
                dLoss__d_outputs_cross_entropy_times_weights += np.sum(d_loss_times_linear_weights[2].item())

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
                gpu.append(res.gpu)
                gpu_mem.append(res.memory)
                '''all_writer.add_scalar('GPU/gpu', res.gpu, time_stamp)
                all_writer.add_scalar('GPU/gpu-mem', res.memory, time_stamp)'''

            if args.time:
                time_full_minibatch = time.time() - time_start
                all_writer.add_scalar('Time/Full Minibatch [Sec]', time_full_minibatch, time_stamp)

        all_writer.add_scalar('Train/Nan count', nan_count, e)

        if DEVICE.type == 'cuda':
            all_writer.add_scalar('GPU/gpu', np.mean(gpu), e)
            all_writer.add_scalar('GPU/gpu-mem', np.mean(gpu_mem), e)

        if args.time:
            time_start_performance_calc = time.time()

        # Compute C index:
        # PAY ATTENTION: the function 'concordance_index_censored' takes censored = True as not censored (This is why we should invert 'all_censored' )
        # and takes the outputs as a risk NOT as survival time !!!!!!!!!!
        if args.loss_combination:
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

            fpr_train_Cox, tpr_train_Cox, _ = roc_curve(relevant_binary_targets, relevant_outputs_Cox)
            roc_auc_train_Cox = auc(fpr_train_Cox, tpr_train_Cox)

            fpr_train_L2, tpr_train_L2, _ = roc_curve(relevant_binary_targets, relevant_outputs_L2)
            roc_auc_train_L2 = auc(fpr_train_L2, tpr_train_L2)

            # When working with the Binary model than all data without binary target should be sorted out.
            fpr_train_Binary, tpr_train_Binary, _ = roc_curve(relevant_binary_targets,
                                                              np.array(all_outputs_Binary)[valid_binary_target_indices])
            roc_auc_train_Binary = auc(fpr_train_Binary, tpr_train_Binary)

            all_writer.add_scalars('Train/Loss Per Epoch', {'Total': train_loss / samples_per_epoch,
                                                            'Cox': loss_cox_total / samples_per_epoch,
                                                            'L2': loss_L2_total / samples_per_epoch,
                                                            'Binary': loss_cross_entropy_total / samples_per_epoch
                                                            }, e)

            all_writer.add_scalars('Train/Gradients', {'Total': dLoss__d_outputs,
                                                       'Cox': dLoss__d_outputs_cox,
                                                       'L2': dLoss__d_outputs_L2,
                                                       'Binary': dLoss__d_outputs_cross_entropy
                                                       }, e)

            all_writer.add_scalars('Train/Loss Gradients X Linear Layer', {'Cox': dLoss__d_outputs_cox_times_weights,
                                                                           'L2': dLoss__d_outputs_L2_times_weights,
                                                                           'Binary': dLoss__d_outputs_cross_entropy_times_weights
                                                                           }, e)

            all_writer.add_scalars('Train/C-index', {'Cox': c_index_Cox,
                                                     'L2': c_index_L2,
                                                     'Binary': c_index_Binary
                                                     }, e)

            all_writer.add_scalars('Train/AUC', {'Cox': roc_auc_train_Cox,
                                                 'L2': roc_auc_train_L2,
                                                 'Binary': roc_auc_train_Binary
                                                 }, e)

            print('Loss: {},  Mean C-index: {}, Mean AUC: {}'.format(train_loss,
                                                                     (c_index_Cox + c_index_L2 + c_index_Binary) / 3,
                                                                     (roc_auc_train_Cox + roc_auc_train_L2 + roc_auc_train_Binary) / 3))


        else:
            all_outputs_for_c_index = -np.array(all_outputs) if flip_outputs else all_outputs
            c_index, num_concordant_pairs, num_discordant_pairs, _, _ = concordance_index_censored(np.invert(all_censored),
                                                                                                   all_target_time,
                                                                                                   all_outputs_for_c_index
                                                                                                   )

            # Compute AUC:
            # We compute AUC w.r.t the binary targets so we need to sort out samples with binary target of -1

            all_valid_indices = np.where(np.array(all_target_binary) != -1)[0]
            relevant_binary_targets = np.array(all_target_binary)[all_valid_indices]

            # When using Cox model the scores represent Risk so we should invert them for computing AUC
            all_outputs = -np.array(all_outputs) if (args.target == 'Time' and args.loss == 'Cox') else np.array(all_outputs)
            relevant_outputs = all_outputs[all_valid_indices]

            fpr_train, tpr_train, _ = roc_curve(relevant_binary_targets, relevant_outputs)
            auc_train = auc(fpr_train, tpr_train)

            #print('(Train) C-index: {}, AUC: {}'.format(c_index, auc_train))
            #print(len(scores_xl), len(slides_xl), len(targets_binary_xl), len(targets_time_xl), len(censored_xl))

            # The loss is normalized per sample for each minibatch. We sum all loss for each minibatch so if we want to get
            # the mean loss per sample we need to divide train_loss by the number of minibatches
            all_writer.add_scalar('Train/Loss Per Sample (Per Epoch)', train_loss / size_minibatch, e)
            all_writer.add_scalar('Train/C-index Per Epoch', c_index, e)
            all_writer.add_scalar('Train/AUC Per Epoch', auc_train, e)
            all_writer.add_scalar('Train/d_Loss/d_outputs Per Epoch', dLoss__d_outputs, e)

            print('(Train) Loss: {}, C-index: {}, AUC: {}'.format(train_loss, c_index, auc_train))

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


        if e % args.save_every == 0:
            # Run validation:
            test(e, test_loader)

            utils.run_data(experiment=experiment, epoch=e)  # Update epoch number
            # Save model to file:
            try:
                model_state_dict = model.module.state_dict()
            except AttributeError:
                model_state_dict = model.state_dict()

            torch.save({'epoch': e,
                        'model_state_dict': model_state_dict
                        },
                       os.path.join(args.output_dir, 'Model_CheckPoints', 'model_data_Epoch_' + str(e) + '.pt'))


def test(current_epoch, test_data_loader):
    model.eval()
    model.to(DEVICE)

    if args.loss_combination:
        all_outputs_Cox, all_outputs_L2, all_outputs_Binary = [], [], []

    all_targets_time, all_targets_binary, all_outputs, all_censored = [], [], [], []
    test_loss = 0

    with torch.no_grad():
        for batch_idx, minibatch in enumerate(test_data_loader):

            '''if type(test_data_loader) != torch.utils.data.dataloader.DataLoader:
                minibatch = utils.concatenate_minibatch(minibatch, is_shuffle=False)'''

            data = minibatch['Data']
            target_time = minibatch['Time Target']
            target_binary = minibatch['Binary Target']
            censored = minibatch['Censored']

            all_targets_time.extend(target_time.numpy())
            all_targets_binary.extend(target_binary.numpy())
            all_censored.extend(censored.numpy())

            data = data.to(DEVICE)
            target_binary = target_binary.to(DEVICE)
            target_time = target_time.to(DEVICE)

            target_time = target_time.to(DEVICE)

            outputs, _ = model(data)

            if args.loss_combination:
                all_outputs_Cox.extend(outputs.detach().cpu().numpy()[:, 0])
                all_outputs_L2.extend(outputs.detach().cpu().numpy()[:, 1])
                outputs_for_binary = torch.nn.functional.softmax(outputs[:, 2:], dim=1)
                all_outputs_Binary.extend(outputs_for_binary[:, 1].detach().cpu().numpy())

                loss, _, _, _ = criterion(outputs,
                                          targets_time=target_time,
                                          targets_binary=target_binary,
                                          censored=censored,
                                          weights=args.loss_weights)

            else:
                if args.target == 'Time':
                    loss = criterion(outputs, target_time, censored)
                    all_outputs.extend(outputs.detach().cpu().numpy()[:, 0])

                elif args.target == 'Binary':
                    # For the Binary Target case, we can compute loss only for samples with valid binary target (not '-1')
                    valid_indices = np.where(target_binary.detach().cpu().numpy() != -1)[0]
                    loss = criterion(outputs[valid_indices], target_binary[valid_indices].squeeze(1))

                    outputs_after_sftmx = torch.nn.functional.softmax(outputs, dim=1)
                    all_outputs.extend(outputs_after_sftmx[:, 1].detach().cpu().numpy())

            test_loss += loss.item()

        if args.loss_combination:
            # Compute C index:
            c_index_Cox, _, _, _, _ = concordance_index_censored(np.invert(all_censored),
                                                                 all_targets_time,
                                                                 all_outputs_Cox)
            c_index_L2, _, _, _, _ = concordance_index_censored(np.invert(all_censored),
                                                                all_targets_time,
                                                                -np.array(all_outputs_L2))
            c_index_Binary, _, _, _, _ = concordance_index_censored(np.invert(all_censored),
                                                                    all_targets_time,
                                                                    -np.array(all_outputs_Binary))

            # Compute AUC:
            valid_binary_target_indices = np.where(np.array(all_targets_binary) != -1)[0]
            relevant_binary_targets = np.array(all_targets_binary)[valid_binary_target_indices]

            # When using Cox model the scores represent Risk so we should invert them for computing AUC
            all_outputs_Cox = -np.array(all_outputs_Cox)
            relevant_outputs_Cox = np.array(all_outputs_Cox)[valid_binary_target_indices]
            relevant_outputs_L2 = np.array(all_outputs_L2)[valid_binary_target_indices]

            fpr_test_Cox, tpr_test_Cox, _ = roc_curve(relevant_binary_targets, relevant_outputs_Cox)
            auc_test_Cox = auc(fpr_test_Cox, tpr_test_Cox)

            fpr_test_L2, tpr_test_L2, _ = roc_curve(relevant_binary_targets, relevant_outputs_L2)
            auc_test_L2 = auc(fpr_test_L2, tpr_test_L2)

            fpr_test_Binary, tpr_test_Binary, _ = roc_curve(relevant_binary_targets,
                                                            np.array(all_outputs_Binary)[valid_binary_target_indices])
            auc_test_Binary = auc(fpr_test_Binary, tpr_test_Binary)

            print('Validation set Performance. Mean C-index: {}, Mean AUC: {}'.format((c_index_Cox + c_index_L2 + c_index_Binary) / 3,
                                                                                      (auc_test_Cox + auc_test_L2 + auc_test_Binary) / 3))

            all_writer.add_scalar('Test/Loss Per Epoch', test_loss / len(test_data_loader.dataset), current_epoch)
            all_writer.add_scalars('Test/C-index', {'Cox': c_index_Cox,
                                                    'L2': c_index_L2,
                                                    'Binary': c_index_Binary
                                                    }, current_epoch)
            all_writer.add_scalars('Test/AUC', {'Cox': auc_test_Cox,
                                                'L2': auc_test_L2,
                                                'Binary': auc_test_Binary
                                                }, current_epoch)

        else:
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

            print('Validation set Performance. C-index: {}, AUC: {}'.format(c_index, auc_test))

            all_writer.add_scalar('Test/Loss Per Sample (Per Epoch)', test_loss / len(test_data_loader), current_epoch)
            all_writer.add_scalar('Test/C-index Per Epoch', c_index, current_epoch)
            all_writer.add_scalar('Test/AUC Per Epoch', auc_test, current_epoch)

    model.train()


def save_debug_data(outputs, features):
    print('Saving debug data and exiting...')
    # Saving data for debugging:

    # Gradients for features
    DF = pd.DataFrame(features.grad.detach().cpu().numpy())
    DF.to_excel('DBG_feature_gradients.xlsx')

    # Gradients for weights of layer
    DF = pd.DataFrame(model.linear.weight.grad.detach().cpu().numpy())
    DF.to_excel('DBG_weight_gradients.xlsx')

    # Gradients for Bias of last layer
    DF = pd.DataFrame(model.linear.bias.grad.detach().cpu().numpy())
    DF.to_excel('DBG_bias_gradients.xlsx')

    # Features
    DF = pd.DataFrame(features.detach().cpu().numpy())
    DF.to_excel('DBG_features.xlsx')

    # Weights of last layer:
    DF = pd.DataFrame(model.linear.weight.detach().cpu().numpy())
    DF.to_excel('DBG_weights.xlsx')

    # Bias of last layer
    DF = pd.DataFrame(model.linear.bias.detach().cpu().numpy())
    DF.to_excel('DBG_bias.xlsx')

    # Loss gradients:
    DF = pd.DataFrame(outputs.grad.detach().cpu().numpy())
    DF.to_excel('DBG_loss_gradients.xlsx')

    exit()

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
    if args.loss_combination:
        print('Train with Combined loss with weights: {}'.format(args.loss_weights))
        receptor = 'Survival_Combined_Loss'
    else:
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
                                          weight_decay=args.weight_decay,
                                          censored_ratio=args.censored_ratio,
                                          combined_loss_weights=args.loss_weights_as_str if args.loss_combination else [])

        args.output_dir, experiment = run_data_results['Location'], run_data_results['Experiment']

    else:
        run_data_output = utils.run_data(experiment=args.experiment)
        args.output_dir, args.test_fold, args.transform_type, TILE_SIZE, tiles_per_bag, \
        args.batch_size, args.dx, args.dataset, args.target, args.model, args.mag = \
            run_data_output['Location'], run_data_output['Test Fold'], run_data_output['Transformations'], \
            run_data_output['Tile Size'], run_data_output['Tiles Per Bag'], run_data_output['Num Bags'],\
            run_data_output['DX'], run_data_output['Dataset Name'], run_data_output['Receptor'],\
            run_data_output['Model Name'], run_data_output['Desired Slide Magnification']

        if sys.platform == 'darwin' and 'home' in args.output_dir.split('/'):
            args.output_dir = os.path.join('/Users/wasserman/Developer/WSI_MIL/runs/', args.output_dir.split('/')[5])

        if args.target == 'Survival_Binary':
            args.target = 'Binary'

        elif args.target == 'Survival_Combined_Loss':
            pass  # Don't change the receptor/ target

        else:  #elif args.target in ['Features_Survival_Time_Cox', 'Survival_Time_Cox', 'Survival_Time_L2']:
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

    if args.loss_combination:
        model.linear = nn.Linear(model.linear.in_features, 4)
        criterion = Combined_loss

    elif args.target == 'Time':
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

    train_dset = WSI_REGdataset_Survival_CR(train=True,
                                            DataSet='ABCTB',
                                            tile_size=TILE_SIZE,
                                            target_kind='survival',
                                            test_fold=args.test_fold,
                                            transform_type=args.transform_type
                                            )

    test_dset = WSI_REGdataset_Survival_CR(train=False,
                                           DataSet='ABCTB',
                                           tile_size=TILE_SIZE,
                                           target_kind='survival',
                                           test_fold=args.test_fold,
                                           transform_type='none'
                                           )

    sampler = None
    do_shuffle = True
    if args.censored_ratio > 0 and args.censored_ratio < 1:
        censored_balancing_DF = pd.DataFrame(train_dset.censored * train_dset.factor)
        num_censor_True = np.sum(censored_balancing_DF == True).item()
        num_censor_False = np.sum(censored_balancing_DF == False).item()
        weights = pd.DataFrame(np.zeros(len(train_dset)))
        weights[np.array(censored_balancing_DF == True)] = 1 / num_censor_True
        weights[np.array(censored_balancing_DF == False)] = 1 / num_censor_False
        do_shuffle = False  # the sampler shuffles
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=weights.squeeze(), num_samples=len(train_dset))

    train_loader = DataLoader(train_dset, batch_size=args.mini_batch_size, shuffle=do_shuffle, num_workers=num_workers, pin_memory=True, sampler=sampler)

    test_loader = DataLoader(test_dset, batch_size=args.mini_batch_size * 2, shuffle=False, num_workers=num_workers, pin_memory=True)

    if args.experiment == 0:
        # Save transformation data to 'run_data.xlsx'
        transformation_string = ', '.join([str(train_dset.transform.transforms[i]) for i in range(len(train_dset.transform.transforms))])
        utils.run_data(experiment=experiment, transformation_string=transformation_string)

        # Save model data and data-set size to run_data.xlsx file:
        utils.run_data(experiment=experiment, model=model.model_name)
        utils.run_data(experiment=experiment, DataSet_size=(train_dset.real_length, test_dset.real_length))
        utils.run_data(experiment=experiment, DataSet_Slide_magnification=train_dset.desired_magnification)

        # Saving code files, args and main file name (this file) to Code directory within the run files.
        utils.save_code_files(args, train_dset)

    all_writer = SummaryWriter(os.path.join(args.output_dir, 'writer'))

    if not os.path.isdir(os.path.join(args.output_dir, 'Model_CheckPoints')):
        Path(os.path.join(args.output_dir, 'Model_CheckPoints')).mkdir(parents=True)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train(from_epoch=from_epoch, epochs=args.epochs, data_loader=train_loader)


    all_writer.close()
    print('Done')

