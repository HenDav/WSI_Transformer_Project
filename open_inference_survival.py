from sklearn.metrics import auc, roc_curve, roc_auc_score
from matplotlib import pyplot as plt
import pickle
from cycler import cycler
import numpy as np
import pandas as pd
import os
from sksurv.metrics import concordance_index_censored
from glob import glob
import itertools

#from inference_loader_input import inference_files, inference_dir, save_csv, patient_level, inference_name, dataset, target

custom_cycler = (cycler(color=['#377eb8', '#ff7f00', '#4daf4a',
                                    '#f781bf', '#a65628', '#984ea3',
                                    '#999999', '#e41a1c', '#dede00']) +
                      cycler(linestyle=['solid', 'dashed', 'dotted',
                                        'dashdot', 'solid', 'dashed',
                                        'dotted', 'dashdot', 'dashed']))


def compute_C_index(all_tile_scores, all_slide_time_targets, all_slide_names, all_slide_censored, train_type):
    flip_outputs = True if train_type in ['L2', 'Binary'] else False

    all_scores_for_c_index = -all_tile_scores if flip_outputs else all_tile_scores

    # Computing C-index per tile:
    tile_censor_status = np.repeat(all_slide_censored, all_tile_scores.shape[1])
    tile_time_targets = np.repeat(all_slide_time_targets, all_tile_scores.shape[1])
    tile_scores = np.reshape(all_scores_for_c_index, all_scores_for_c_index.shape[0] * all_scores_for_c_index.shape[1])

    valid_indices = np.where(np.isnan(tile_scores) == False)
    c_index_per_tile, _, _, _, _ = concordance_index_censored(np.invert(tile_censor_status[valid_indices]),
                                                              tile_time_targets[valid_indices],
                                                              tile_scores[valid_indices]
                                                              )

    # Computing C-index per Slide:
    mean_slide_scores, slide_time_targets = [], []
    patient_data = {}
    patients_with_multiple_time_targets = []
    total_patients = 0
    slides_data_DF = pd.read_excel('/Users/wasserman/Developer/WSI_MIL/Data_from_gipdeep/ABCTB_TIF/slides_data_ABCTB.xlsx')
    slides_data_DF.set_index('file', inplace=True)

    for slide_idx in range(all_scores_for_c_index.shape[0]):
        slide_scores = all_scores_for_c_index[slide_idx]
        valid_slide_indices = np.where(np.isnan(slide_scores) == False)
        mean_slide_scores.append(slide_scores[valid_slide_indices].mean())
        slide_time_targets.append(all_slide_time_targets[slide_idx])

        # Gather data Per Patient:
        patient_id = slides_data_DF.loc[all_slide_names[slide_idx]]['patient barcode']
        # Check if patient is in the list of patients with inconsistent targets:
        if patient_id in patients_with_multiple_time_targets:
            continue

        if patient_id in patient_data.keys():
            # Check Target and Censor consistency:
            current_slide_time_target = all_slide_time_targets[slide_idx]
            patients_target = patient_data[patient_id]['Time Target']

            current_slide_censor_status = all_slide_censored[slide_idx]
            patients_censor_status = patient_data[patient_id]['Censor']

            if current_slide_time_target != patients_target or current_slide_censor_status != patients_censor_status:
                patients_with_multiple_time_targets.append(patient_id)
                del patient_data[patient_id]
                continue

            else:
                patient_data[patient_id]['Scores'].extend(slide_scores[valid_slide_indices].tolist())

        else:
            total_patients += 1
            patient_data[patient_id] = {'Scores': slide_scores[valid_slide_indices].tolist(),
                                        'Time Target': all_slide_time_targets[slide_idx],
                                        'Censor': all_slide_censored[slide_idx]}

    c_index_per_slide, _, _, _, _ = concordance_index_censored(np.invert(all_slide_censored),
                                                               slide_time_targets,
                                                               mean_slide_scores
                                                               )

    # Computing C-index per Patient:
    patient_scores, patient_time_targets, patient_censor = [], [], []
    for patient_id in patient_data.keys():
        patient_scores.append(np.mean(patient_data[patient_id]['Scores']))
        patient_time_targets.append(patient_data[patient_id]['Time Target'])
        patient_censor.append(patient_data[patient_id]['Censor'])

    c_index_per_patient, _, _, _, _ = concordance_index_censored(np.invert(patient_censor),
                                                               patient_time_targets,
                                                               patient_scores
                                                               )

    return {'Tile': c_index_per_tile,
            'Slide': c_index_per_slide,
            'Patient': c_index_per_patient
            }


def compute_AUC(all_tile_scores, all_slide_targets, all_slide_names, train_type, c_index_values:dict = {'Tile': -1, 'Slide': -1, 'Patient': -1}):
    if train_type == 'Cox':
        all_tile_scores = -all_tile_scores
    # Computing AUC per tile:
    valid_indices = np.where(all_slide_targets >= 0)[0]
    valid_tile_scores = all_tile_scores[valid_indices]
    valid_targets = all_slide_targets[valid_indices]
    augmented_valid_targets = np.repeat(valid_targets, 500)
    augmented_tile_scores = np.reshape(valid_tile_scores, valid_tile_scores.shape[0] * valid_tile_scores.shape[1])

    # Sorting out Nan scores:
    valid_indices = np.where(np.isnan(augmented_tile_scores) == False)[0]
    augmented_valid_targets = augmented_valid_targets[valid_indices]
    augmented_tile_scores = augmented_tile_scores[valid_indices]

    fpr_tile, tpr_tile, _ = roc_curve(augmented_valid_targets, augmented_tile_scores)
    AUC_tile = auc(fpr_tile, tpr_tile)

    # Computing AUC per Slide and per patient:
    mean_slide_scores, slide_targets = [], []
    patient_data = {}
    patients_with_multiple_targets = []
    total_patients = 0
    slides_data_DF = pd.read_excel('/Users/wasserman/Developer/WSI_MIL/Data_from_gipdeep/ABCTB_TIF/slides_data_ABCTB.xlsx')
    slides_data_DF.set_index('file', inplace=True)

    for slide_idx in range(valid_tile_scores.shape[0]):
        slide_scores = valid_tile_scores[slide_idx]
        valid_slide_indices = np.where(np.isnan(slide_scores) == False)
        mean_slide_scores.append(slide_scores[valid_slide_indices].mean())
        slide_targets.append(valid_targets[slide_idx])

        # Gather data Per Patient:
        patient_id = slides_data_DF.loc[all_slide_names[slide_idx]]['patient barcode']
        # Check if patient is in the list of patients with inconsistent targets:
        if patient_id in patients_with_multiple_targets:
            continue

        if patient_id in patient_data.keys():
            # Check Target consistency:
            current_slides_target = valid_targets[slide_idx].item()
            patients_target = patient_data[patient_id]['Target']
            if current_slides_target != patients_target:
                patients_with_multiple_targets.append(patient_id)
                del patient_data[patient_id]
                continue

            else:
                patient_data[patient_id]['Scores'].extend(slide_scores[valid_slide_indices].tolist())

        else:
            total_patients += 1
            patient_data[patient_id] = {'Scores': slide_scores[valid_slide_indices].tolist(),
                                        'Target': valid_targets[slide_idx].item()}

    fpr_slide, tpr_slide, _ = roc_curve(slide_targets, mean_slide_scores)
    AUC_slide = auc(fpr_slide, tpr_slide)

    # Computing AUC per Patient:
    patient_scores, patient_targets = [], []
    for patient_id in patient_data.keys():
        patient_scores.append(np.mean(patient_data[patient_id]['Scores']))
        patient_targets.append(patient_data[patient_id]['Target'])

    fpr_patient, tpr_patient, _ = roc_curve(patient_targets, patient_scores)
    AUC_patient = auc(fpr_patient, tpr_patient)

    # Plot AUC:
    fig1, ax1 = plt.subplots()
    ax1.set_prop_cycle(custom_cycler)
    ax1.plot(fpr_tile, tpr_tile)
    ax1.plot(fpr_slide, tpr_slide)
    ax1.plot(fpr_patient, tpr_patient)
    legend_labels = ['Tile AUC: ' + str(round(AUC_tile, 4)) + ', C-index: ' + str(round(c_index_values['Tile'], 4)),
                     'Slide AUC: ' + str(round(AUC_slide, 4)) + ', C-index: ' + str(round(c_index_values['Slide'], 4)),
                     'Patient AUC: ' + str(round(AUC_patient, 4)) + ', C-index: ' + str(round(c_index_values['Patient'], 4))]
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(legend_labels)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(b=True)
    plt.title('AUC for ' + train_type + ' Loss, Removed {}/{} Patients'.format(len(patients_with_multiple_targets), total_patients))
    plt.savefig(os.path.join(file_location, train_type + '_AUC_Model_' + model + '_Epoch_' + epoch + '.png'))
    plt.close()

    return {'Tile': (fpr_tile, tpr_tile, AUC_tile),
            'Slide': (fpr_slide, tpr_slide, AUC_slide),
            'Patient': (fpr_patient, tpr_patient, AUC_patient)
            }


def compute_L2_loss(all_tile_scores, all_slide_time_targets, all_slide_censored, train_type):
    if train_type == 'Cox':
        all_tile_scores = -all_tile_scores

    tile_time_targets = np.array(list(itertools.chain.from_iterable([[time_target] * all_tile_scores.shape[1] for time_target in all_slide_time_targets])))
    tile_censored = np.array(list(itertools.chain.from_iterable([[censor] * all_tile_scores.shape[1] for censor in all_slide_censored])))
    tile_scores = np.reshape(all_tile_scores, (all_tile_scores.size, ))

    valid_indices_1 = np.where(tile_censored == False)[0]
    valid_indices_2 = np.where(tile_scores < tile_time_targets)[0]
    valid_indices = list(set(np.concatenate((valid_indices_1, valid_indices_2))))
    loss = np.sum((tile_scores[valid_indices] - tile_time_targets[valid_indices]) ** 2)

    print('L2 Loss for {} Train is: {}'.format(train_type, np.sqrt(loss)))


def _save_slide_scores_and_targets_to_csv_for_one_score(slide_names, slide_targets, tile_scores, slide_dataset, loss_type):
    patch_scores_df = pd.DataFrame(tile_scores)
    patch_scores_df.insert(0, "slide_name", slide_names)
    patch_scores_df.insert(0, "dataset", slide_dataset)

    if type(slide_targets) == dict:
        for target in slide_targets.keys():
            patch_scores_df.insert(0, target + ' Target', slide_targets[target])

    patch_scores_df.to_csv(os.path.join(file_location,
                                        'Model_{}_Epoch_{}_LossType_{}_tile_scores.csv'.format(model, epoch, loss_type)))


def save_slide_scores_and_targets_to_csv(slide_names, slide_targets, tile_scores, slide_dataset, loss_type):
    if len(tile_scores.shape) == 2:
        _save_slide_scores_and_targets_to_csv_for_one_score(slide_names=slide_names,
                                                            slide_targets=slide_targets,
                                                            tile_scores=tile_scores,
                                                            slide_dataset=slide_dataset,
                                                            loss_type=loss_type
                                                            )

    elif len(tile_scores.shape) == 3 and tile_scores.shape[2] == 4:
        for score_idx in range(3):
            if score_idx <= 1:
                current_tile_scores = tile_scores[:, :, score_idx]
            elif score_idx == 2:
                current_tile_scores = tile_scores[:, :, 3]

            _save_slide_scores_and_targets_to_csv_for_one_score(slide_names=slide_names,
                                                                slide_targets=slide_targets,
                                                                tile_scores=current_tile_scores,
                                                                slide_dataset=slide_dataset,
                                                                loss_type=loss_type[score_idx])



    else:
        raise Exception('tile_scores is not in the required shape')


def get_loss_type(file_name):
    if 'Binary' in file_name:
        return 'Binary'
    elif 'Cox' in file_name:
        return 'Cox'
    elif 'L2' in file_name:
        return 'L2'
    elif 'Combined_Loss' in file_name:
        return ['Cox', 'L2', 'Binary']
    else:
        raise Exception('Unrecognized Loss type')



if __name__ == '__main__':
    #file_name = '/Users/wasserman/Developer/WSI_MIL/runs/Exp_10609-Survival_Binary-TestFold_1/Inference/Model_Epoch_1000-Folds_1_survival-Tiles_500.data'
    #file_name = '/Users/wasserman/Developer/WSI_MIL/runs/Exp_10607-Survival_Time_Cox-TestFold_1/Inference/Model_Epoch_960-Folds_1_survival-Tiles_500.data'
    #file_name = '/Users/wasserman/Developer/WSI_MIL/runs/Exp_10611-Survival_Time_L2-TestFold_1/Inference/Model_Epoch_1000-Folds_1_survival-Tiles_500.data'
    file_location = '/Users/wasserman/Developer/WSI_MIL/runs/Exp_10645-Survival_Combined_Loss-TestFold_1/Inference/'
    files = glob(os.path.join(file_location, '*.data'))

    folder = file_location
    model = files[0].split('/')[6].split('_')[1].split('-')[0]

    AUC_all_models_data = {}

    for file_name in files:
        epoch = file_name.split('/')[8].split('_')[2].split('-')[0]
        loss_type = get_loss_type(file_name)
        with open(file_name, 'rb') as filehandle:
            inference_data = pickle.load(filehandle)

        if len(inference_data) == 13:  # survival format
            fpr, tpr, all_scores_after_softmax, NUM_SLIDES, tile_scores_after_softmax, all_slide_names,\
            all_slide_datasets, tile_locations, binary_targets, time_targets, censor_status,\
            all_scores_before_softmax, tile_scores_before_softmax = inference_data

        else:
            IOError('inference data is of unsupported size!')

        save_slide_scores_and_targets_to_csv(slide_names=all_slide_names,
                                             slide_targets={'Time': time_targets,
                                                            'Binary': binary_targets
                                                            },
                                             tile_scores=tile_scores_after_softmax,
                                             slide_dataset=all_slide_datasets,
                                             loss_type=loss_type)



        if 'Cox' in file_name:
            train_type = 'Cox'
        elif 'L2' in file_name:
            train_type = 'L2'
        elif 'Binary' in file_name:
            train_type = 'Binary'
        elif 'Survival_Combined_Loss' in file_name:
            train_type = 'Combined_Loss'
        else:
            raise Exception('Train type cannot be found')

        if train_type == 'Combined_Loss':
            # COX
            c_index_values_COX = compute_C_index(all_tile_scores=tile_scores_after_softmax[:, :, 0],
                                                 all_slide_time_targets=time_targets,
                                                 all_slide_names=all_slide_names,
                                                 all_slide_censored=censor_status,
                                                 train_type='Cox')

            auc_data_COX = compute_AUC(all_tile_scores=tile_scores_after_softmax[:, :, 0],
                                       all_slide_targets=binary_targets,
                                       all_slide_names=all_slide_names,
                                       train_type='Cox',
                                       c_index_values=c_index_values_COX)

            compute_L2_loss(all_tile_scores=tile_scores_after_softmax[:, :, 0],
                                          all_slide_time_targets=time_targets,
                                          all_slide_censored=censor_status,
                                          train_type='Cox')
            # L2
            c_index_values_L2 = compute_C_index(all_tile_scores=tile_scores_after_softmax[:, :, 1],
                                                all_slide_time_targets=time_targets,
                                                all_slide_names=all_slide_names,
                                                all_slide_censored=censor_status,
                                                train_type='L2')

            auc_data_L2 = compute_AUC(all_tile_scores=tile_scores_after_softmax[:, :, 1],
                                      all_slide_targets=binary_targets,
                                      all_slide_names=all_slide_names,
                                      train_type='L2',
                                      c_index_values=c_index_values_L2)

            compute_L2_loss(all_tile_scores=tile_scores_after_softmax[:, :, 1],
                                         all_slide_time_targets=time_targets,
                                         all_slide_censored=censor_status,
                                         train_type='L2')

            # BINARY
            c_index_values_BINARY = compute_C_index(all_tile_scores=tile_scores_after_softmax[:, :, 3],
                                                    all_slide_time_targets=time_targets,
                                                    all_slide_names=all_slide_names,
                                                    all_slide_censored=censor_status,
                                                    train_type='Binary')

            auc_data_BINARY = compute_AUC(all_tile_scores=tile_scores_after_softmax[:, :, 3],
                                          all_slide_targets=binary_targets,
                                          all_slide_names=all_slide_names,
                                          train_type='Binary',
                                          c_index_values=c_index_values_BINARY)

            c_index_collection = {'Cox': c_index_values_COX,
                                  'L2': c_index_values_L2,
                                  'Binary': c_index_values_BINARY}

            AUC_collection = {'Cox': auc_data_COX,
                              'L2': auc_data_L2,
                              'Binary': auc_data_BINARY}

            AUC_all_models_data[epoch] = AUC_collection

        else:
            c_index_values = compute_C_index(all_tile_scores=tile_scores_after_softmax,
                                             all_slide_time_targets=time_targets,
                                             all_slide_names=all_slide_names,
                                             all_slide_censored=censor_status,
                                             train_type=train_type)

            auc_data = compute_AUC(all_tile_scores=tile_scores_after_softmax,
                                   all_slide_targets=binary_targets,
                                   all_slide_names=all_slide_names,
                                   train_type=train_type,
                                   c_index_values=c_index_values)


            AUC_all_models_data[epoch] = auc_data

    # Plot AUC:
    fig_tile, ax_tile = plt.subplots()
    ax_tile.set_prop_cycle(custom_cycler)
    fig_slide, ax_slide = plt.subplots()
    ax_slide.set_prop_cycle(custom_cycler)
    fig_patient, ax_patient = plt.subplots()
    ax_patient.set_prop_cycle(custom_cycler)

    legend_labels_tile, legend_labels_slide, legend_labels_patient = [], [], []

    for key in AUC_all_models_data.keys():
        if train_type == 'Combined_Loss':
            for loss_type in AUC_all_models_data[key].keys():
                fpr_tile, tpr_tile, AUC_tile = AUC_all_models_data[key][loss_type]['Tile']
                fpr_slide, tpr_slide, AUC_slide = AUC_all_models_data[key][loss_type]['Slide']
                fpr_patient, tpr_patient, AUC_patient = AUC_all_models_data[key][loss_type]['Patient']

                ax_tile.plot(fpr_tile, tpr_tile)
                ax_slide.plot(fpr_slide, tpr_slide)
                ax_patient.plot(fpr_patient, tpr_patient)

                legend_labels_tile.append(key + ' ' + loss_type + ': ' + str(round(AUC_tile, 4)))
                legend_labels_slide.append(key + ' ' + loss_type + ': ' + str(round(AUC_slide, 4)))
                legend_labels_patient.append(key + ' ' + loss_type + ': ' + str(round(AUC_patient, 4)))

        else:
            fpr_tile, tpr_tile, AUC_tile = AUC_all_models_data[key]['Tile']
            fpr_slide, tpr_slide, AUC_slide = AUC_all_models_data[key]['Slide']
            fpr_patient, tpr_patient, AUC_patient = AUC_all_models_data[key]['Patient']

            ax_tile.plot(fpr_tile, tpr_tile)
            ax_slide.plot(fpr_slide, tpr_slide)
            ax_patient.plot(fpr_patient, tpr_patient)

            legend_labels_tile.append(key + ': ' + str(round(AUC_tile, 4)))
            legend_labels_slide.append(key + ': ' + str(round(AUC_slide, 4)))
            legend_labels_patient.append(key + ': ' + str(round(AUC_patient, 4)))

    commands = ['plt.figure(1)', 'plt.figure(2)', 'plt.figure(3)']
    legends = [legend_labels_tile, legend_labels_slide, legend_labels_patient]
    titles = ['Tile_AUC', 'Slide_AUC', 'Patient_AUC']
    for idx in range(len(commands)):
        eval(commands[idx])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(b=True)

        plt.legend(legends[idx])
        plt.title(titles[idx])
        plt.savefig(os.path.join(file_location, titles[idx] + '_Comparison.png'))
        plt.close()

    print('Done')


