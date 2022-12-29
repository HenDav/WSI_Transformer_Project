from sksurv.metrics import concordance_index_censored
from Omer_files_suspected_as_unnecessary.omer_datasets import Features_to_Survival
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
from itertools import cycle
from random import shuffle
from Nets.PreActResNets import PreActResNet50_Ron
import copy
import time

torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(description='')
parser.add_argument('-ex', '--experiment', type=int, default=0, help='Continue train of this experiment')
parser.add_argument('-fe', '--from_epoch', type=int, default=0, help='Continue train from epoch')
parser.add_argument('-e', '--epochs', default=501, type=int, help='Epochs to run')
parser.add_argument('-tar', '--target', type=str, default='Binary', help='Binary / Time')
parser.add_argument('-l', '--loss', type=str, default='Cox', help='Cox / L2')
parser.add_argument('-mb', '--mini_batch_size', type=int, default=20, help='Mini batch size')
#parser.add_argument('-wc', '--without_censored', dest='without_censored', action='store_true', help='train without censpred data')
parser.add_argument('-time', dest='time', action='store_true', help='save train timing data ?')
parser.add_argument('-grads', dest='grads', action='store_true', help='save model gradients data ?')
parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=0.03, type=float, help='L2 penalty')
parser.add_argument('--eps', default=1e-5, type=float, help='epsilon (for optimizer')
parser.add_argument('-cr', '--censored_ratio', type=float, default=-0.5, help='ratio of censored samples in each minibatch')
parser.add_argument('-pf', dest='patient_features', action='store_false', help='use patient features ?')
parser.add_argument('-nf', dest='normalized_features', action='store_true', help='use patient features ?')
args = parser.parse_args()


if args.normalized_features == True and args.patient_features == True:
    raise Exception('Only one of the flags normalized_features and patient_features can be set to TRUE')


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
    if args.grads:
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
            start_time_minibatch = time.time()
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

            if args.time:
                all_writer.add_scalar('Time/Mean time to get data from Dataset ', minibatch['Time'].mean().item(), time_stamp)

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
            if args.grads:
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
                valid_indices = np.where(target_binary.detach().cpu().numpy() != -1)[0]
                outputs = outputs[valid_indices]
                target_binary = target_binary[valid_indices]
                if args.grads:
                    outputs.retain_grad()
                loss = criterion(outputs, target_binary)

            train_loss += loss.item()

            loss.backward()
            if args.grads:
                dLoss__d_outputs += np.sum(np.abs(outputs.grad.detach().cpu().numpy()))
            optimizer.step()

            if DEVICE.type == 'cuda':
                res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
                all_writer.add_scalar('GPU/gpu', res.gpu, time_stamp)
                all_writer.add_scalar('GPU/gpu-mem', res.memory, time_stamp)

            if args.grads:
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

            if args.time:
                MB_time = time.time() - start_time_minibatch
                all_writer.add_scalar('Time/Minibatch time', MB_time, time_stamp)
                all_writer.add_scalar('Time/Minibatch time X minibatch len', MB_time * minibatch_len, time_stamp)


        if 'scheduler' in locals():
            scheduler.step()

        if args.time:
            start_time_epoch_performance = time.time()
        # Compute C index:
        # PAY ATTENTION: the function 'concordance_index_censored' takes censored = True as not censored (This is why we should invert 'all_censored' )
        # and takes the outputs as a risk NOT as survival time !!!!!!!!!!
        all_outputs_for_c_index = -np.array(all_outputs) if flip_outputs else all_outputs
        #print(np.invert(all_censored).shape, all_target_time.shape, all_outputs_for_c_index.shape)
        c_index, _, _, _, _ = concordance_index_censored(np.invert(all_censored),
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

        if args.time:
            end_time_epoch_performance = time.time()
            all_writer.add_scalar('Time/Epoch preformance', end_time_epoch_performance - start_time_epoch_performance, e)

        if e % 20 == 0:
            if args.time:
                start_time_test = time.time()
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
                       os.path.join(args.output_dir, 'Model_CheckPoints', 'model_data_Epoch_' + str(e) + '.pt'))

            if args.time:
                end_time_test = time.time()
                all_writer.add_scalar('Time/Test time', end_time_test - start_time_test, e)

    all_writer.close()


def test(current_epoch, test_data_loader):
    # is_per_patient_dataset = True if test_data_loader.dataset.is_per_patient or test_data_loader.dataset.is_per_slide else False

    model.eval()
    model.to(DEVICE)

    all_targets_time, all_targets_binary, all_outputs, all_censored = [], [], [], []
    all_targets_time_per_slide, all_targets_binary_per_slide, all_outputs_per_slide, all_censored_per_slide = [], [], [], []
    all_data_per_patient = {}
    #all_targets_time_per_patient, all_targets_binary_per_patient, all_outputs_per_patient, all_censored_per_patient = [], [], [], []
    # all_patient_ids_per_patient = []
    all_patient_ids, all_patient_ids_per_slide = [], []
    test_loss = 0
    test_loss_per_slide = 0

    with torch.no_grad():
        for batch_idx, minibatch in enumerate(test_data_loader):
            data = minibatch['Features']
            if len(data.shape) == 3 and np.where(np.array(data.shape) == 1)[0].size != 0:
                location_to_squeeze = np.where(np.array(data.shape) == 1)[0][0]
                data = data.squeeze(location_to_squeeze)

            target_time = minibatch['Time Target'].item()
            target_binary = minibatch['Binary Target']
            censored = minibatch['Censored'].item()
            patient_id = minibatch['Slide Belongs to Patient'][0]


            all_targets_time.extend([target_time] * data.size(0))
            all_targets_binary.extend([target_binary.numpy()] * data.size(0))
            all_censored.extend([censored] * data.size(0))
            all_patient_ids.extend([patient_id] * data.size(0))

            all_targets_time_per_slide.append(target_time)
            all_targets_binary_per_slide.append(target_binary.numpy())
            all_censored_per_slide.append(censored)
            all_patient_ids_per_slide.append(patient_id)

            # Create a new patient in the dict:
            if patient_id not in test_data_loader.dataset.bad_patient_list and patient_id not in all_data_per_patient.keys():
                all_data_per_patient[patient_id] = {'Time Target': target_time,
                                                    'Binary Target': target_binary,
                                                    'Censor': censored,
                                                    'Outputs': []}

            data = data.to(DEVICE)
            target_binary = target_binary.to(DEVICE)

            outputs = model(data)

            if args.target == 'Time':
                '''if is_per_patient_dataset:
                    outputs = outputs.mean()
                    all_outputs.append(outputs.item())

                else:'''
                all_outputs.extend(outputs.detach().cpu().numpy()[:, 0])
                all_outputs_per_slide.append(outputs.mean().item())  # TODO: Debug this line

                if patient_id not in test_data_loader.dataset.bad_patient_list:
                    all_data_per_patient[patient_id]['Outputs'].extend(outputs.detach().cpu().numpy()[:, 0])

            elif args.target == 'Binary':
                outputs_after_sftmx = torch.nn.functional.softmax(outputs, dim=1)

                all_outputs.extend(outputs_after_sftmx[:, 1].detach().cpu().numpy())
                outputs_after_sftmx_mean = outputs_after_sftmx.mean(0)  # Computing mean output for all tiles
                all_outputs_per_slide.append(outputs_after_sftmx_mean[1].detach().cpu().numpy())

                if patient_id not in test_data_loader.dataset.bad_patient_list:
                    all_data_per_patient[patient_id]['Outputs'].extend(outputs_after_sftmx[:, 1].detach().cpu().numpy())

                '''if is_per_patient_dataset:
                    outputs = outputs_after_sftmx.mean(0)  # Computing mean output for all tiles
                    all_outputs.append(outputs[1].detach().cpu().numpy())
                    

                    if target_binary == -1:  # If the binary target for this slide/patient is -1 we cannot use it for loss computation
                        continue

                    # Finishing the computation for CrossEntropy we need to log the softmaxed outputs and then do NLLLoss
                    loss = nn.functional.nll_loss(torch.log(outputs).reshape(1, 2), torch.LongTensor([target_binary]))

                else:'''
                if target_binary != -1:
                    target_binary_multiplied = target_binary.detach().cpu() * torch.ones(outputs.size(0))
                    target_binary_multiplied = target_binary_multiplied.to(torch.long)
                    loss = criterion(outputs, target_binary_multiplied.to(DEVICE))
                    test_loss += loss.item()

                    # compute per slide loss:
                    # Finishing the computation for CrossEntropy we need to log the softmaxed outputs and then do NLLLoss
                    loss_per_slide = nn.functional.nll_loss(torch.log(outputs_after_sftmx_mean).reshape(1, 2), torch.LongTensor([target_binary]).to(DEVICE))
                    test_loss_per_slide += loss_per_slide.item()


    # Compute loss for Time target after collecting all the data will avoid problems that might occur due to to much censored data.
    #test_loss = criterion(torch.tensor(all_outputs).reshape(len(all_outputs), 1), torch.tensor(all_targets_time), torch.tensor(all_censored))
    if args.target == 'Time':
        # Per Tile
        test_loss = criterion(torch.reshape(torch.tensor(all_outputs), (len(all_outputs), 1)), torch.tensor(all_targets_time), torch.tensor(all_censored))
        # Per Slide
        test_loss_per_slide = criterion(torch.reshape(torch.tensor(all_outputs_per_slide), (len(all_outputs_per_slide), 1)), torch.tensor(all_targets_time_per_slide), torch.tensor(all_censored_per_slide))


    # Compute C index:
    all_outputs_for_c_index = -np.array(all_outputs) if flip_outputs else all_outputs
    c_index, _, _, _, _ = concordance_index_censored(np.invert(all_censored),
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
    auc_test = auc(fpr_test, tpr_test)

    string_init = 'Test/Per Tile/'

    item_in_dset = len(test_data_loader.dataset)

    num_MB = len(test_data_loader) if args.target == 'Binary' else 1


    print('Validation set Performance (Per Tile). C-index: {}, AUC: {}, Loss(per {} num of MiniBatch): {}'.format(c_index, auc_test, item_in_dset, test_loss / num_MB))

    all_writer.add_scalar(string_init + 'Loss Per MiniBatch (Per Epoch)', test_loss / num_MB, current_epoch)
    all_writer.add_scalar(string_init + 'C-index Per Epoch', c_index, current_epoch)
    all_writer.add_scalar(string_init + 'AUC Per Epoch', auc_test, current_epoch)

    model.train()

    # Compute performance per slide and per patient
    compute_per_patient_and_slide_performance(all_outputs_per_slide, all_censored_per_slide, all_targets_time_per_slide, all_targets_binary_per_slide, test_loss_per_slide, all_data_per_patient, current_epoch=current_epoch)


def compute_per_patient_and_slide_performance(scores, censor_data, time_targets, binary_targets, test_loss_slide, all_patient_data, current_epoch):
    # Compute C index Per Slide:
    all_outputs_for_c_index = -np.array(scores) if flip_outputs else scores
    c_index, _, _, _, _ = concordance_index_censored(np.invert(censor_data),
                                                     time_targets,
                                                     all_outputs_for_c_index
                                                     )

    # Compute AUC:
    all_valid_indices = np.where(np.array(binary_targets) != -1)[0]
    relevant_binary_targets = np.array(binary_targets)[all_valid_indices]

    # When using Cox model the scores represent Risk so we should invert them for computing AUC
    all_outputs = -np.array(scores) if (args.target == 'Time' and args.loss == 'Cox') else scores
    relevant_outputs = np.array(all_outputs)[all_valid_indices]

    fpr_test, tpr_test, _ = roc_curve(relevant_binary_targets, relevant_outputs)
    auc_test = auc(fpr_test, tpr_test)

    string_init = 'Test/Per Slide/'
    num_slides = len(scores) if args.target == 'Binary' else 1
    print('Validation set Performance (Per Slide). C-index: {}, AUC: {}, Loss(per {} num of MiniBatch): {}'.format(c_index,
                                                                                                                   auc_test,
                                                                                                                   num_slides,
                                                                                                                   test_loss_slide / num_slides))

    all_writer.add_scalar(string_init + 'Loss Per MiniBatch (Per Epoch)', test_loss_slide / num_slides, current_epoch)
    all_writer.add_scalar(string_init + 'C-index Per Epoch', c_index, current_epoch)
    all_writer.add_scalar(string_init + 'AUC Per Epoch', auc_test, current_epoch)

    #####################################################################################
    # Compute performance per patient.
    # First we'll go over all patients and computer their mean scores
    scores_patient, time_targets_patient, binary_targets_patient, censor_patient = [], [], [], []
    for patient in all_patient_data.keys():
        scores_patient.append(np.array(all_patient_data[patient]['Outputs']).mean())
        time_targets_patient.append(all_patient_data[patient]['Time Target'])
        binary_targets_patient.append(all_patient_data[patient]['Binary Target'])
        censor_patient.append(all_patient_data[patient]['Censor'])


    # Computing C index:
    scores_patient_for_c_index = -np.array(scores_patient) if flip_outputs else scores_patient
    c_index, _, _, _, _ = concordance_index_censored(np.invert(censor_patient),
                                                     time_targets_patient,
                                                     scores_patient_for_c_index
                                                     )

    # Compute AUC:
    all_valid_indices = np.where(np.array(binary_targets_patient) != -1)[0]
    relevant_binary_targets = np.array(binary_targets_patient)[all_valid_indices]

    # When using Cox model the scores represent Risk so we should invert them for computing AUC
    all_outputs_patient = -np.array(scores_patient) if (args.target == 'Time' and args.loss == 'Cox') else scores_patient
    relevant_outputs = np.array(all_outputs_patient)[all_valid_indices]

    fpr_test, tpr_test, _ = roc_curve(relevant_binary_targets, relevant_outputs)
    auc_test = auc(fpr_test, tpr_test)

    # Compute Loss per patient:
    if args.target == 'Time':
        loss_per_patient = criterion(torch.reshape(torch.tensor(scores_patient), (len(scores_patient), 1)), torch.tensor(time_targets_patient), torch.tensor(censor_patient))
    elif args.target == 'Binary':
        loss_per_patient = nn.functional.nll_loss(torch.log(torch.tensor((1 - relevant_outputs, relevant_outputs))).reshape(len(relevant_outputs), 2), torch.LongTensor(relevant_binary_targets))

    string_init = 'Test/Per Patient/'
    print('Validation set Performance (Per Patient). C-index: {}, AUC: {}, Loss(per {} num of MiniBatch): {}'.format(c_index,
                                                                                                                   auc_test,
                                                                                                                   1,
                                                                                                                   loss_per_patient))

    all_writer.add_scalar(string_init + 'Loss Per MiniBatch (Per Epoch)', test_loss_slide / num_slides, current_epoch)
    all_writer.add_scalar(string_init + 'C-index Per Epoch', c_index, current_epoch)
    all_writer.add_scalar(string_init + 'AUC Per Epoch', auc_test, current_epoch)


########################################################################################################################
########################################################################################################################

if __name__ == '__main__':

    DEVICE = utils.device_gpu_cpu()
    num_workers = utils.get_cpu()

    if DEVICE.type == 'cuda':
        import nvidia_smi
        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

    # Model definition:
    num_added_features = 17 if args.patient_features else 0
    if args.target == 'Time':
        model = nn.Linear(512 + num_added_features, 1)
        if args.loss == 'Cox':
            criterion = Cox_loss
            flip_outputs = False
            #main_dir = 'Test_Run/Features/Time_Cox'
        elif args.loss == 'L2':
            criterion = L2_Loss
            flip_outputs = True
            #main_dir = 'Test_Run/Features/Time_L2'
        else:
            Exception('No valid loss function defined')

    elif args.target == 'Binary':
        model = nn.Linear(512 + num_added_features, 2)
        criterion = nn.CrossEntropyLoss()
        flip_outputs = True
        #main_dir = 'Test_Run/Features/Binary'

    if args.normalized_features:
        receptor = 'NormalizedFeatures_Survival_'
    elif args.patient_features:
        receptor = 'AugmentedFeatures_Survival_'
    else:
        receptor = 'Features_Survival_'

    receptor += args.target + '_WD_' + str(args.weight_decay)
    if args.target == 'Time':
        receptor += '_' + args.loss

    if args.experiment == 0:
        run_data_results = utils.run_data(test_fold=1,
                                          transform_type='none',
                                          tile_size=0,
                                          tiles_per_bag=1,
                                          DataSet_name='FEATURES_Augmented: Survival',
                                          Receptor=receptor,
                                          num_bags=args.mini_batch_size,
                                          learning_rate=args.lr)

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
        elif args.target == 'Survival_Time':
            args.target = 'Time'

        print('args.dataset:', args.dataset)
        print('args.target:', args.target)
        print('args.args.batch_size:', args.batch_size)
        print('args.output_dir:', args.output_dir)
        print('args.test_fold:', args.test_fold)
        print('args.transform_type:', args.transform_type)
        print('args.dx:', args.dx)

        experiment = args.experiment
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


    # Initializing Datasets
    test_set = Features_to_Survival(is_train=False, normalized_features=args.normalized_features, use_patient_features=args.patient_features)

    if args.censored_ratio < 0:  # We're using the data as is and not manipulating the amount of censored/not censored samples in each minibatch
        train_set = Features_to_Survival(is_train=True, normalized_features=args.normalized_features, use_patient_features=args.patient_features)

    elif args.censored_ratio == 0:  # All samples are NOT censored
        train_set = Features_to_Survival(is_train=True, normalized_features=args.normalized_features, use_patient_features=args.patient_features, is_all_not_censored=True)

    elif args.censored_ratio > 0 and args.censored_ratio < 1:
        train_set_censored = Features_to_Survival(is_train=True, normalized_features=args.normalized_features, use_patient_features=args.patient_features, is_all_censored=True)
        train_set_not_censored = Features_to_Survival(is_train=True, normalized_features=args.normalized_features, use_patient_features=args.patient_features, is_all_not_censored=True)

    # Initializing DataLoaders:
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=num_workers)

    if args.censored_ratio <= 0:
        train_loader = {'Censored': DataLoader(train_set, batch_size=args.mini_batch_size, shuffle=True, num_workers=num_workers),
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

    # Saving code files, args and main file name (this file) to Code directory within the run files.
    utils.save_code_files(args, None)

    if not os.path.isdir(os.path.join(args.output_dir, 'Model_CheckPoints')):
        Path(os.path.join(args.output_dir, 'Model_CheckPoints')).mkdir(parents=True)

    all_writer = SummaryWriter(os.path.join(args.output_dir, 'writer'))

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

