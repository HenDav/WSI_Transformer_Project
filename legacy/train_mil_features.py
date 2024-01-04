import utils
import datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch
import torch.optim as optim

from Nets.PreActResNets import PreActResNet50_Ron
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
from sklearn.metrics import roc_curve, auc
import numpy as np
import sys
import copy

import utils_MIL
from Nets import nets_mil

# utils.send_run_data_via_mail()

parser = argparse.ArgumentParser(description='WSI_MIL Training of PathNet Project')
parser.add_argument('-ds', '--dataset', type=str, default='TCGA_ABCTB', help='DataSet to use')
parser.add_argument('-tar', '--target', type=str, default='ER', help='Target to train for') #FIXME: to Her2+is_Tumor
parser.add_argument('-tf', '--test_fold', default=2, type=int, help='fold to be as VALIDATION FOLD, if -1 there is no validation. refered to as TEST FOLD in folder hiererchy and code. very confusing, I agree.')
parser.add_argument('-e', '--epochs', default=2, type=int, help='Epochs to run')
parser.add_argument('-ex', '--experiment', type=int, default=0, help='Continue train of this experiment')
parser.add_argument('-fe', '--from_epoch', type=int, default=0, help='Continue train from epoch')
parser.add_argument('-ppt', dest='per_patient_training', action='store_true', help='will the data be taken per patient (or per slides) ?')
parser.add_argument('-time', dest='time', action='store_true', help='save train timing data ?')
parser.add_argument('-nb', '--num_bags', type=int, default=50, help='Number of bags in each minibatch')#FIXME: to 50
parser.add_argument('-tpb', '--tiles_per_bag', type=int, default=100, help='Tiles Per Bag') #FIXME: to 100
parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=5e-5, type=float, help='L2 penalty')
parser.add_argument('--model', default='nets_mil.MIL_Feature_Attention_MultiBag()', type=str, help='net to use')
parser.add_argument('--eval_rate', type=int, default=5, help='Evaluate validation set every # epochs')
parser.add_argument('-llf', dest='last_layer_freeze', action='store_true', help='get last layer and freeze it ?')
parser.add_argument('-dl', '--data_limit', type=int, default=None, help='Data Limit to a specified number of feature tiles')
parser.add_argument('-repData', dest='repeating_data', action='store_false', help='sample data with repeat ?')
parser.add_argument('-conly', dest='carmel_only', action='store_true', help='Use ONLY CARMEL slides  ?')
parser.add_argument('-remark', '--remark', type=str, default='', nargs=argparse.REMAINDER, help='option to add remark for the run')
parser.add_argument('--is_tumor_mode', type=int, default=0, help='which mode to use when training with target receptor + is_tumor')

args = parser.parse_args()

if not (isinstance(args.is_tumor_mode, int) and args.is_tumor_mode in [0, 1, 2]):
    raise Exception('args.is_tumor_mode must be one of 0/1/2')

EPS = 1e-7

def train(model: nn.Module, dloader_train: DataLoader, dloader_test: DataLoader, DEVICE, optimizer, print_timing: bool=False):
    """
    This function trains the model
    :return:
    """
    writer_folder = os.path.join(args.output_dir, 'writer')
    all_writer = SummaryWriter(os.path.join(writer_folder, 'all'))
    if print_timing:
        time_writer = SummaryWriter(os.path.join(writer_folder, 'time'))

    if from_epoch == 0:
        all_writer.add_text('Experiment No.', str(experiment))
        all_writer.add_text('Train type', 'MIL')
        all_writer.add_text('Model type', str(type(model)))

    print()
    print('Training will be conducted with {} bags and {} tiles per bag in each MiniBatch'.format(args.num_bags, args.tiles_per_bag))
    print('Start Training...')
    previous_epoch_loss = 1e5

    for e in range(from_epoch, epoch + from_epoch):
        time_epoch_start = time.time()

        # The following 3 lines initialize variables to compute AUC for train dataset.
        total_train, correct_pos_train, correct_neg_train = 0, 0, 0
        total_pos_train, total_neg_train = 0, 0
        true_targets_train, scores_train = np.zeros(0), np.zeros(0)
        correct_labeling, train_loss = 0, 0

        print('Epoch {}:'.format(e))
        model.train()
        for batch_idx, minibatch in enumerate(tqdm(dloader_train)):
            target = minibatch['targets']
            data = minibatch['features']
            if '+is_Tumor' in args.target:
                # conctenating both data vectors:
                data = torch.cat((data, minibatch['tumor_features']), axis=2)

            train_start = time.time()

            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            model.to(DEVICE)

            outputs, weights, _ = model(x=None, H=data)

            if str(outputs.min().item()) == 'nan':
                print('slides:', minibatch['slide name'])
                print('features', data.shape)
                print('feature 1:', data[0].min().item(), data[0].max().item())
                print('feature 2:', data[1].min().item(), data[1].max().item())
                print('num tiles:', minibatch['num tiles'])
                print('feature1', data[0])
                print('feature2', data[1])

                exit()

            weights = weights.cpu().detach().numpy()

            loss = criterion(outputs, target)
            train_loss += loss.item()

            outputs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            loss.backward()
            optimizer.step()

            scores_train = np.concatenate((scores_train, outputs[:, 1].cpu().detach().numpy()))

            true_targets_train = np.concatenate((true_targets_train, target.cpu().detach().numpy()))

            total_train += target.size(0)
            total_pos_train += target.eq(1).sum().item()
            total_neg_train += target.eq(0).sum().item()
            correct_labeling += predicted.eq(target).sum().item()

            correct_pos_train += predicted[target.eq(1)].eq(1).sum().item()
            correct_neg_train += predicted[target.eq(0)].eq(0).sum().item()

            # Calculate training accuracy
            all_writer.add_scalar('Loss', loss.item(), batch_idx + e * len(dloader_train))

            train_time = time.time() - train_start

            if print_timing:
                time_stamp = batch_idx + e * len(dloader_train)
                time_writer.add_scalar('Time/Train (iter) [Sec]', train_time, time_stamp)

        time_epoch = (time.time() - time_epoch_start) / 60
        if print_timing:
            time_writer.add_scalar('Time/Full Epoch [min]', time_epoch, e)

        train_acc = 100 * correct_labeling / total_train
        balanced_acc_train = 100 * (correct_pos_train / (total_pos_train + EPS) + correct_neg_train / (total_neg_train + EPS)) / 2

        fpr_train, tpr_train, _ = roc_curve(true_targets_train, scores_train)
        roc_auc_train = auc(fpr_train, tpr_train)

        all_writer.add_scalar('Train/Balanced Accuracy', balanced_acc_train, e)
        all_writer.add_scalar('Train/Roc-Auc', roc_auc_train, e)
        all_writer.add_scalar('Train/Loss Per Epoch', train_loss, e)
        all_writer.add_scalar('Train/Accuracy', train_acc, e)
        all_writer.add_scalar('Train/Weights mean Total (per bag)', np.mean(np.sum(weights, axis=1)), e)
        all_writer.add_scalar('Train/Weights mean Variance (per bag)', np.mean(np.var(weights, axis=1)), e)

        print('Finished Epoch: {}, Loss: {:.2f}, Loss Delta: {:.3f}, Train AUC per patch: {:.2f} , Time: {:.0f} m'
              .format(e,
                      train_loss,
                      previous_epoch_loss - train_loss,
                      roc_auc_train,
                      time_epoch))

        previous_epoch_loss = train_loss

        if e % args.eval_rate == 0:
            if len(dloader_test) != 0:
                acc_test, bacc_test = check_accuracy(model, dloader_test, all_writer, DEVICE, e)
            else:
                acc_test, bacc_test = None, None
                
            # Update 'Last Epoch' at run_data.xlsx file:
            utils.run_data(experiment=experiment, epoch=e)

            # Save model to file:
            if not os.path.isdir(os.path.join(args.output_dir, 'Model_CheckPoints')):
                os.mkdir(os.path.join(args.output_dir, 'Model_CheckPoints'))
            try:
                model_state_dict = model.module.state_dict()
            except AttributeError:
                model_state_dict = model.state_dict()

            torch.save({'epoch': e,
                        'model_state_dict': model_state_dict,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss.item(),
                        'acc_test': acc_test,
                        'bacc_test': bacc_test,
                        'tiles_per_bag': args.tiles_per_bag},
                       os.path.join(args.output_dir, 'Model_CheckPoints', 'model_data_Epoch_' + str(e) + '.pt'))

    all_writer.close()
    if print_timing:
        time_writer.close()


def check_accuracy(model: nn.Module, data_loader: DataLoader, writer_all, DEVICE, epoch: int):
    test_loss, total_test = 0, 0
    correct_labeling_test = 0
    total_pos_test, total_neg_test = 0, 0
    correct_pos_test, correct_neg_test = 0, 0
    targets_test, scores_test = np.zeros(0), np.zeros(0)

    model.eval()

    with torch.no_grad():
        for idx, minibatch_val in enumerate(data_loader):
            target = minibatch_val['targets']
            data = minibatch_val['features']
            if '+is_Tumor' in args.target:
                # conctenating both data vectors:
                data = torch.cat((data, minibatch_val['tumor_features']), axis=2)

            data, target = data.to(device=DEVICE), target.to(device=DEVICE)
            model.to(DEVICE)

            outputs, weights, _ = model(x=None, H=data)

            outputs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            scores_test = np.concatenate((scores_test, outputs[:, 1].cpu().detach().numpy()))
            targets_test = np.concatenate((targets_test, target.cpu().detach().numpy()))

            total_test += target.size(0)
            correct_labeling_test += predicted.eq(target).sum().item()
            total_pos_test += target.eq(1).sum().item()
            total_neg_test += target.eq(0).sum().item()
            correct_pos_test += predicted[target.eq(1)].eq(1).sum().item()
            correct_neg_test += predicted[target.eq(0)].eq(0).sum().item()

        acc = 100 * float(correct_labeling_test) / total_test
        balanced_acc = 100 * (correct_pos_test / (total_pos_test + EPS) + correct_neg_test / (total_neg_test + EPS)) / 2

        fpr, tpr, _ = roc_curve(targets_test, scores_test)
        roc_auc = auc(fpr, tpr)

        writer_all.add_scalar('Test/Accuracy', acc, epoch)
        writer_all.add_scalar('Test/Balanced Accuracy', balanced_acc, epoch)
        writer_all.add_scalar('Test/Roc-Auc', roc_auc, epoch)

        print('Slide AUC of {:.2f} over Test set'.format(roc_auc))

    model.train()
    return acc, balanced_acc

##################################################################################################


if __name__ == '__main__':
    # Device definition:
    DEVICE = utils.device_gpu_cpu()

    # Get number of available CPUs:
    cpu_available = utils.get_cpu()

    # Data type definition:
    DATA_TYPE = 'Features'
    data_location = utils_MIL.get_RegModel_Features_location_dict(train_DataSet=args.dataset, target=args.target, test_fold=args.test_fold)
    

    # TODO: WHAT DOES THIS DO ?
    if sys.platform == 'darwin' and type(data_location) == tuple and (data_location[0]['TrainSet Location'] == None and data_location[1]['TrainSet Location'] == None):
        data_location[0]['TrainSet Location'] = data_location[0]['TestSet Location']
        data_location[1]['TrainSet Location'] = data_location[1]['TestSet Location']

    # Saving/Loading run meta data to/from file:
    if args.experiment == 0:
        run_data_results = utils.run_data(test_fold=args.test_fold,
                                          transform_type=None,
                                          tile_size=0,
                                          tiles_per_bag=args.tiles_per_bag,
                                          num_bags=args.num_bags,
                                          DX=None,
                                          DataSet_name=data_location[0]['DataSet Name'] + ' + IS_TUMOR' if type(data_location) == tuple else data_location['DataSet Name'],
                                          is_per_patient=args.per_patient_training,
                                          is_last_layer_freeze=args.last_layer_freeze,
                                          is_repeating_data=args.repeating_data,
                                          Receptor=args.target + '_Features',
                                          MultiSlide=True,
                                          learning_rate=args.lr,
                                          weight_decay=args.weight_decay,
                                          DataSet_Slide_magnification=0,
                                          data_limit=args.data_limit,
                                          carmel_only=args.carmel_only,
                                          Remark=' '.join(args.remark),
                                          receptor_tumor_mode=args.is_tumor_mode)

        args.output_dir, experiment = run_data_results['Location'], run_data_results['Experiment']
    else:
        run_data_output = utils.run_data(experiment=args.experiment)
        args.output_dir, args.test_fold, args.transformation, TILE_SIZE, args.tiles_per_bag, args.num_bags, args.dx, \
        args.dataset, args.target, is_MultiSlide, args.model, args.mag =\
            run_data_output['Location'], run_data_output['Test Fold'], run_data_output['Transformations'], run_data_output['Tile Size'],\
            run_data_output['Tiles Per Bag'], run_data_output['Num Bags'], run_data_output['DX'], run_data_output['Dataset Name'],\
            run_data_output['Receptor'], run_data_output['MultiSlide'], run_data_output['Model Name'], run_data_output['Desired Slide Magnification']

        experiment = args.experiment


    # Fix target:
    if args.target == 'ER_for_is_Tumor':
        args.target = 'ER'

    # Get data:
    if 'OR' in args.target:
        train_dset = datasets.Features_MILdataset_combined(dataset=args.dataset,
                                              data_location=data_location['TrainSet Location'],
                                              is_per_patient=args.per_patient_training,
                                              is_repeating_tiles=args.repeating_data,
                                              bag_size=args.tiles_per_bag,
                                              target=args.target,
                                              is_train=True,
                                              data_limit=args.data_limit,
                                              test_fold=args.test_fold,
                                              carmel_only=args.carmel_only)
        test_dset = datasets.Features_MILdataset_combined(dataset=args.dataset,
                                             data_location=data_location['TestSet Location'],
                                             is_per_patient=args.per_patient_training,
                                             bag_size=args.tiles_per_bag,
                                             target=args.target,
                                             is_train=False,
                                             test_fold=args.test_fold,
                                             carmel_only=args.carmel_only)
    else:
        train_dset = datasets.Features_MILdataset(dataset=args.dataset,
                                              data_location=(data_location[0]['TrainSet Location'], data_location[1]['TrainSet Location']) if type(data_location) == tuple else data_location['TrainSet Location'],
                                              is_per_patient=args.per_patient_training,
                                              is_repeating_tiles=args.repeating_data,
                                              bag_size=args.tiles_per_bag,
                                              target=args.target,
                                              is_train=True,
                                              data_limit=args.data_limit,
                                              test_fold=args.test_fold,
                                              carmel_only=args.carmel_only)
    
        test_dset = datasets.Features_MILdataset(dataset=args.dataset,
                                             data_location=(data_location[0]['TestSet Location'], data_location[1]['TestSet Location']) if type(data_location) == tuple else data_location['TestSet Location'],
                                             is_per_patient=args.per_patient_training,
                                             bag_size=args.tiles_per_bag,
                                             target=args.target,
                                             is_train=False,
                                             test_fold=args.test_fold,
                                             carmel_only=args.carmel_only)

    train_loader = DataLoader(train_dset, batch_size=args.num_bags, shuffle=True, num_workers=cpu_available, pin_memory=True)
    test_loader = DataLoader(test_dset, batch_size=args.num_bags, shuffle=False, num_workers=cpu_available, pin_memory=True)

    # Load model
    model = eval(args.model)
    if args.experiment != 0:  # In case we continue from an already trained model, than load the previous model and optimizer data:
        print('Loading pre-saved model...')
        model_data_loaded = torch.load(os.path.join(args.output_dir,
                                                    'Model_CheckPoints',
                                                    'model_data_Epoch_' + str(args.from_epoch) + '.pt'),
                                       map_location='cpu')

        model.load_state_dict(model_data_loaded['model_state_dict'])

        from_epoch = args.from_epoch + 1
        print()
        print('Resuming training of Experiment {} from Epoch {}'.format(args.experiment, args.from_epoch))

    elif args.last_layer_freeze:  # This part will load the last linear layer from the REG model into the last layer (classifier part) of the attention module

        if type(data_location) == tuple:
            print('Copying and freezeing last layer from model \"{}\"'.format(data_location[0]['REG Model Location']))
            basic_model_data = torch.load(data_location[0]['REG Model Location'], map_location='cpu')['model_state_dict']
        else:
            print('Copying and freezeing last layer from model \"{}\"'.format(data_location['REG Model Location']))
            basic_model_data = torch.load(data_location['REG Model Location'], map_location='cpu')['model_state_dict']

        basic_model = PreActResNet50_Ron()
        basic_model.load_state_dict(basic_model_data)

        last_linear_layer_data = copy.deepcopy(basic_model.linear.state_dict())
        model.classifier.load_state_dict(last_linear_layer_data)

        for p in model.classifier.parameters():  # This part will freeze the classifier part so it won't change during training
            p.requires_grad = False

    if model.model_name in ['nets_mil.MIL_Feature_Attention_MultiBag()',
                            'nets_mil.MIL_Feature_2_Attention_MultiBag()',
                            'nets_mil.MIL_Feature_3_Attention_MultiBag()']:
        model.tiles_per_bag = args.tiles_per_bag

        if '+is_Tumor' in args.target:
            model.set_isTumor_train_mode(isTumor_train_mode=args.is_tumor_mode)

    # Save model data and DataSet size (and some other dataset data) to run_data.xlsx file (Only if this is a new run).
    if args.experiment == 0:
        utils.run_data(experiment=experiment, model=model.model_name, DataSet_size=(len(train_dset), len(test_dset)))

        # Saving code files, args and main file name (this file) to Code directory within the run files.
        utils.save_code_files(args, train_dset)

    epoch = args.epochs
    from_epoch = args.from_epoch

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if DEVICE.type == 'cuda':
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()

    train(model, train_loader, test_loader, DEVICE=DEVICE, optimizer=optimizer, print_timing=args.time)
    print('Training No. {} has concluded successfully after {} Epochs'.format(experiment, args.epochs))
