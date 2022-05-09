from sklearn.metrics import auc, roc_curve, roc_auc_score
from matplotlib import pyplot as plt
import pickle
from cycler import cycler
import numpy as np
import pandas as pd
import os
from inference_loader_input import inference_files, inference_dir, save_csv, patient_level, inference_name, dataset, target

multi_target = False
if len(target.split('+')) > 1:
    multi_target = True
    #target0, target1 = target.split('+')
    #target_names = [target0, target1]
    target_list = target.split('+')
    N_targets = len(target_list)

custom_cycler = (cycler(color=['#377eb8', '#ff7f00', '#4daf4a',
                                    '#f781bf', '#a65628', '#984ea3',
                                    '#999999', '#e41a1c', '#dede00']) +
                      cycler(linestyle=['solid', 'dashed', 'dotted',
                                        'dashdot', 'solid', 'dashed',
                                        'dotted', 'dashdot', 'dashed']))

infer_type = 'REG'

def auc_for_n_patches(patch_scores, n, all_targets):
    max_n = patch_scores.shape[1]
    n_iter = 10
    auc_array = np.zeros(n_iter)
    for iter in range(n_iter):
        patches = np.random.choice(np.arange(max_n), n, replace=False)
        chosen_patches = patch_scores[:, patches]
        chosen_mean_scores = np.array([np.nanmean(chosen_patches[ii, chosen_patches[ii, :] > 0]) for ii in range(chosen_patches.shape[0])])

        # TODO RanS 4.2.21 - handle slides with nans (less than max_n patches)
        #temp fix - remove slides if all selected patches are nan
        chosen_targets = np.array([all_targets[ii] for ii in range(len(all_targets)) if ~np.isnan(chosen_mean_scores[ii])])
        chosen_mean_scores = np.array([chosen_mean_score for chosen_mean_score in chosen_mean_scores if ~np.isnan(chosen_mean_score)])
        #chosen_targets = np.array([all_targets[patch] for patch in patches])
        auc_array[iter] = roc_auc_score(chosen_targets, chosen_mean_scores)

    auc_res = np.nanmean(auc_array)
    return auc_res


roc_auc = []
slide_score_all = []

if len(inference_files) == 0:
    raise IOError('No inference files found!')

# read is_cancer values from experiment 627, temp RanS 13.3.22
is_cancer_improv = False
if is_cancer_improv:
    is_cancer_inference_path = r'C:\Pathnet_results\MIL_general_try4\BENIGN_runs\is_cancer\exp627\Inference\CAT_Her2_fold1_patches_inference\Model_Epoch_500-Folds_[1]_is_cancer-Tiles_500.data'
    #is_cancer_inference_path = r'C:\Pathnet_results\MIL_general_try4\BENIGN_runs\is_cancer\exp627\Inference\CAT_PR_fold1_patches_inference\Model_Epoch_500-Folds_[1]_is_cancer-Tiles_500.data'
    with open(os.path.join(inference_dir, is_cancer_inference_path), 'rb') as filehandle:
        inference_data_is_cancer = pickle.load(filehandle)
        _, _, _, _, _, _, _, _, _, num_slides_is_cancer, patch_scores_is_cancer, all_slide_names_is_cancer, _, patch_locs_is_cancer = inference_data_is_cancer

for ind, key in enumerate(inference_files.keys()):
    with open(os.path.join(inference_dir, inference_files[key]), 'rb') as filehandle:
        print(key)
        inference_data = pickle.load(filehandle)

    if key[-8:] == 'test_500':
        key = key[:-9]

    if infer_type == 'REG':
        if len(inference_data) == 17: # survival format
            fpr, tpr, all_labels, all_targets, all_scores, total_pos, true_pos, total_neg, true_neg, \
            num_slides, patch_scores, all_slide_names, all_slide_datasets, patch_locs,\
            binary_targets_arr, time_targets_arr, censored_arr = inference_data
        elif len(inference_data) == 14: #current format
            fpr, tpr, all_labels, all_targets, all_scores, total_pos, true_pos, total_neg, true_neg, \
            num_slides, patch_scores, all_slide_names, all_slide_datasets, patch_locs = inference_data
        elif len(inference_data) == 13: #old format, before locations
            fpr, tpr, all_labels, all_targets, all_scores, total_pos, true_pos, total_neg, true_neg, \
            num_slides, patch_scores, all_slide_names, all_slide_datasets = inference_data
        elif len(inference_data) == 12: #old format
            fpr, tpr, all_labels, all_targets, all_scores, total_pos, true_pos, total_neg, true_neg, \
            num_slides, patch_scores, all_slide_names = inference_data
        elif len(inference_data) == 16: #temp old format with patch locs
            fpr, tpr, all_labels,  all_targets, all_scores, total_pos, true_pos, total_neg, true_neg, \
            num_slides, patch_scores, all_slide_names, patch_locs, patch_locs_inds, all_slide_size, all_slide_size_ind = inference_data
        else:
            IOError('inference data is of unsupported size!')


        if is_cancer_improv: #validate patches are the same
            test1 = num_slides_is_cancer == num_slides
            test2 = np.all(all_slide_names == all_slide_names_is_cancer)
            test3 = np.nanmax(patch_locs - patch_locs_is_cancer) == 0
            if (not test1) or (not test2) or (not test3):
                raise IOError('mismatch between is_cancer patches and inference patches')

        if ind == 0: #define figure and axes
            if all_scores.ndim == 2 and target != 'survival' and not multi_target:
                N_classes = all_scores.shape[1]
                fig_list, ax_list, legend_labels = [], [], []
                if patient_level:
                    fig_list_patient, ax_list_patient, legend_labels_patient = [], [], []
                for i_thresh in range(N_classes - 1):
                    fig1, ax1 = plt.subplots()
                    fig_list.append(fig1)
                    ax_list.append(ax1)
                    ax_list[i_thresh].set_prop_cycle(custom_cycler)
                    ax_list[i_thresh].set_title('Threshold ' + str(i_thresh))
                    legend_labels.append([])
                    if patient_level:
                        fig1, ax1 = plt.subplots()
                        fig_list_patient.append(fig1)
                        ax_list_patient.append(ax1)
                        ax_list_patient[i_thresh].set_prop_cycle(custom_cycler)
                        ax_list_patient[i_thresh].set_title('Threshold ' + str(i_thresh))
                        legend_labels_patient.append([])

            else:
                N_classes = 2
                fig1, ax1 = plt.subplots()
                ax1.set_prop_cycle(custom_cycler)
                legend_labels = []
                if patient_level:
                    fig1_patient, ax1_patient = plt.subplots()
                    ax1_patient.set_prop_cycle(custom_cycler)
                    legend_labels_patient = []

        #RanS 9.12.21, auc per dataset
        Temp = False
        if Temp:
            roc_auc1 = roc_auc_score(np.array(all_targets)[all_slide_datasets=='Ipatimup'], np.array(all_scores)[all_slide_datasets=='Ipatimup'])
            roc_auc2 = roc_auc_score(np.array(all_targets)[all_slide_datasets == 'Covilha'],np.array(all_scores)[all_slide_datasets == 'Covilha'])
            roc_auc3 = roc_auc_score(np.array(all_targets)[all_slide_datasets == 'HEROHE'],np.array(all_scores)[all_slide_datasets == 'HEROHE'])

            q = pd.read_excel(r'C:\Pathnet_results\MIL_general_try4\ABCTB_runs\survival\exp20094\Inference\train\survival_fold2, 3, 4, 5_exp20094_epoch1000_patch_scores_fixed.xlsx')
            roc_auc2 = roc_auc_score(np.array(q[q['test fold idx breast'] == 2]['slide_label']), np.array(q[q['test fold idx breast'] == 2]['slide_score']))
            roc_auc3 = roc_auc_score(np.array(q[q['test fold idx breast'] == 3]['slide_label']), np.array(q[q['test fold idx breast'] == 3]['slide_score']))
            roc_auc4 = roc_auc_score(np.array(q[q['test fold idx breast'] == 4]['slide_label']), np.array(q[q['test fold idx breast'] == 4]['slide_score']))
            roc_auc5 = roc_auc_score(np.array(q[q['test fold idx breast'] == 5]['slide_label']), np.array(q[q['test fold idx breast'] == 5]['slide_score']))

        if save_csv:
            if N_classes == 2 and not multi_target:
                patch_scores_df = pd.DataFrame(patch_scores)
                patch_scores_df.insert(0, "slide_name", all_slide_names)
                patch_scores_df.insert(0, "dataset", all_slide_datasets)
                patch_scores_df.insert(0, "slide_label", all_targets)
                patch_scores_df.insert(0, "slide_score", all_scores)
                patch_scores_df.to_csv(os.path.join(inference_dir, key + '_patch_scores.csv'))
            else:
                N_files = np.max((N_targets, N_classes))


                for i_file in range(N_files):
                    patch_scores_df = pd.DataFrame(patch_scores[:, :, i_file])
                    patch_scores_df.insert(0, "slide_name", all_slide_names)
                    patch_scores_df.insert(0, "dataset", all_slide_datasets)
                    if multi_target:
                        patch_scores_df.insert(0, "slide_label", all_targets[i_file,:])
                    else: #multiclass
                        patch_scores_df.insert(0, "slide_label", all_targets)
                    patch_scores_df.insert(0, "slide_score", all_scores[:, i_file])
                    if N_classes > 2:
                        patch_scores_df.to_csv(os.path.join(inference_dir, key + '_patch_scores_class' + str(i_file) +  '.csv'))
                    else:
                        patch_scores_df.to_csv(os.path.join(inference_dir, key + '_patch_scores_' + target_list[i_file] + '.csv'))
            try:
                patch_x_df = pd.DataFrame(patch_locs[:, :, 0])
                patch_x_df.insert(0, "slide_name", all_slide_names)
                patch_x_df.to_csv(os.path.join(inference_dir, key + '_x.csv'))

                patch_y_df = pd.DataFrame(patch_locs[:, :, 1])
                patch_y_df.insert(0, "slide_name", all_slide_names)
                patch_y_df.to_csv(os.path.join(inference_dir, key + '_y.csv'))

                slide_size_df = pd.DataFrame(all_slide_size)
                slide_size_df.insert(0, "slide_name", all_slide_names)
                slide_size_df.to_csv(os.path.join(inference_dir, key + '_slide_dimensions.csv'))
            except:
                pass

        if fpr.__class__ == int and not multi_target:
            my_scores = np.array(all_scores)
            my_targets = np.array(all_targets)
            my_scores = my_scores[my_targets >= 0]
            my_targets = my_targets[my_targets >= 0]
            if N_classes == 2:
                fpr, tpr, _ = roc_curve(my_targets, my_scores)
                roc_auc.append(roc_auc_score(my_targets, my_scores))
            else: #multiclass
                for i_thresh in range(N_classes - 1):
                    roc_auc_per_thresh = np.float64(np.nan)
                    N_pos = np.sum(my_targets >= (i_thresh + 1))  # num of positive targets
                    N_neg = np.sum(my_targets < (i_thresh + 1))  # num of negative targets
                    if (N_pos > 0) and (N_neg > 0):  # more than one label
                        scores_thresh = np.sum(my_scores[:, i_thresh + 1:], axis=1)
                        true_targets_thresh = my_targets >= (i_thresh + 1)
                        fpr, tpr, _ = roc_curve(true_targets_thresh, scores_thresh)
                        roc_auc = auc(fpr, tpr)
                        ax_list[i_thresh].plot(fpr, tpr)
                        legend_labels[i_thresh].append(key + ' (AUC=' + str(round(roc_auc, 3)) + ')')
        else:
            if multi_target:
                for i_target in range(N_targets):
                    my_scores = np.array(all_scores[:, i_target])
                    my_targets = np.array(all_targets[i_target,:])
                    my_scores = my_scores[my_targets >= 0]
                    my_targets = my_targets[my_targets >= 0]
                    fpr, tpr, _ = roc_curve(my_targets, my_scores)
                    try:
                        roc_auc.append(roc_auc_score(my_targets, my_scores))
                        ax1.plot(fpr, tpr)
                        legend_labels.append(key + ' ' + target_list[i_target] + ' (AUC=' + str(round(roc_auc[-1], 3)) + ')')
                    except ValueError: #Only one class present in y_true. ROC AUC score is not defined in that case.
                        pass
            else:
                if is_cancer_improv:
                    #patch_scores_with_is_cancer = patch_scores * patch_scores_is_cancer
                    #tot_score_is_cancer = np.nansum(patch_scores_is_cancer,axis=1).reshape((patch_scores_is_cancer.shape[0],1))
                    #patch_scores_with_is_cancer = patch_scores * patch_scores_is_cancer / tot_score_is_cancer
                    patch_scores_is_cancer1 = patch_scores_is_cancer
                    #patch_scores_is_cancer1[patch_scores_is_cancer1 < 0.995] = np.nan
                    patch_scores_is_cancer1[patch_scores_is_cancer1 < 0.9] = np.nan
                    patch_scores_with_is_cancer = patch_scores * patch_scores_is_cancer1
                    scores_with_is_cancer = np.nanmean(patch_scores_with_is_cancer, axis=1)
                    scores_with_is_cancer[np.isnan(scores_with_is_cancer)] = 0
                    fpr_w_is_cancer, tpr_w_is_cancer, _ = roc_curve(all_targets, scores_with_is_cancer)
                    auc(fpr_w_is_cancer, tpr_w_is_cancer)
                    roc_auc.append(auc(fpr_w_is_cancer, tpr_w_is_cancer))
                else:
                    roc_auc.append(auc(fpr, tpr))
                #plt.plot(fpr, tpr)
                ax1.plot(fpr, tpr)
                legend_labels.append(key + ' (AUC=' + str(round(roc_auc[-1], 3)) + ')')
        # RanS 18.1.21
        #temp fix RanS 4.2.21
        #if patch_scores.ndim == 3:
        #    patch_scores = np.squeeze(patch_scores[:, ind,:])
        # all_scores = np.max(patch_scores, axis=1) #maxpool - temp! RanS 20.1.21
        if N_classes == 2:
            slide_score_mean = np.array([np.nanmean(patch_scores[ii, patch_scores[ii, :] > 0]) for ii in range(patch_scores.shape[0])])
            #slide_score_std = np.array([np.nanstd(patch_scores[ii, patch_scores[ii, :] > 0]) for ii in range(patch_scores.shape[0])])
        else:
            slide_score_mean = all_scores
        slide_score_all.append(slide_score_mean)

        #results per patient
        if patient_level:
            patient_all = []
            if dataset == 'LEUKEMIA':
                slides_data_file = r'C:\ran_data\BoneMarrow\slides_data_LEUKEMIA.xlsx'
            elif (dataset == 'CAT') or (dataset == 'CARMEL'):
                slides_data_file = r'C:\ran_data\Carmel_Slides_examples\add_ki67_labels\ER100_labels\slides_data_CARMEL_labeled_merged.xlsx'
            elif (dataset == 'HAEMEK'):
                slides_data_file = r'C:\ran_data\Haemek\slides_data_HAEMEK1.xlsx'
            elif (dataset == 'SHEBA'):
                slides_data_file = r'C:\ran_data\Sheba\slides_data_SHEBA_labeled_merged.xlsx'
            elif (dataset == 'BENIGN'):
                slides_data_file = r'C:\ran_data\Benign\final slides_data\slides_data_BENIGN_merged.xlsx'

            #temp RanS 27.1.22 for HAEMEK inference on CAT models
            #slides_data_file = r'C:\ran_data\Haemek\slides_data_HAEMEK1.xlsx'
            #slides_data_file = r'C:\ran_data\Carmel_Slides_examples\add_ki67_labels\ER100_labels\slides_data_CARMEL_labeled_merged.xlsx'

            slides_data = pd.read_excel(slides_data_file)

            #for name in all_slide_names:
            for name, slide_dataset in zip(all_slide_names, all_slide_datasets):
                if slide_dataset == 'TCGA': #TCGA files
                    patient_all.append(name[8:12])
                elif slide_dataset == 'ABCTB': #ABCTB files
                    patient_all.append(name[:9])
                elif slide_dataset[:6] == 'CARMEL':  # CARMEL files
                    patient_all.append(slides_data[slides_data['file'] == name]['patient barcode'].item())
                elif dataset == 'LEUKEMIA' or dataset == 'SHEBA':
                    patient_all.append(slides_data[slides_data['file'] == name]['PatientID'].item())
                elif (slide_dataset[:6] == 'HAEMEK') or (slide_dataset[:6] == 'BENIGN'):  # HAEMEK files
                    patient_all.append(slides_data[slides_data['file'] == name]['PatientIndex'].item())

            if N_classes == 2:
                patch_df = pd.DataFrame({'patient': patient_all, 'scores': slide_score_mean, 'targets': all_targets})
            else:
                patch_df = pd.DataFrame({'patient': patient_all, 'targets': all_targets})
                for i_class in range(N_classes):
                    patch_df['scores_' + str(i_class)] = slide_score_mean[:, i_class]

            #RanS 27.1.22, remove patients with multiple targets
            patient_std_df = patch_df.groupby('patient').std()
            invalid_patients = patient_std_df[patient_std_df['targets'] > 0].index
            print('number of patients removed due to multiple targets: ' + str(len(invalid_patients)))
            for invalid_patient in invalid_patients:
                patch_df = patch_df[patch_df.patient != invalid_patient]
            patient_mean_score_df = patch_df.groupby('patient').mean()

            if N_classes == 2:
                #roc_auc_patient = roc_auc_score(patient_mean_score_df['targets'].astype(int), patient_mean_score_df['scores'])
                roc_auc_patient = roc_auc_score(patient_mean_score_df['targets'], patient_mean_score_df['scores'])
                fpr_patient, tpr_patient, _ = roc_curve(patient_mean_score_df['targets'].astype(int),
                                                                         patient_mean_score_df['scores'])
                #plt.plot(fpr_patient, tpr_patient)
                ax1_patient.plot(fpr_patient, tpr_patient)
                legend_labels_patient.append(key + ' (patient AUC=' + str(round(roc_auc_patient, 3)) + ')')
            else:
                my_scores_patient = np.array([patient_mean_score_df['scores_' + str(i_class)] for i_class in range(N_classes)]).T
                for i_thresh in range(N_classes - 1):
                    roc_auc_per_thresh = np.float64(np.nan)
                    N_pos = np.sum(patient_mean_score_df['targets'] >= (i_thresh + 1))  # num of positive targets
                    N_neg = np.sum(patient_mean_score_df['targets'] < (i_thresh + 1))  # num of negative targets
                    if (N_pos > 0) and (N_neg > 0):  # more than one label
                        patient_scores_thresh = np.sum(my_scores_patient[:, i_thresh + 1:], axis=1)
                        patient_true_targets_thresh = patient_mean_score_df['targets'] >= (i_thresh + 1)
                        fpr_patient, tpr_patient, _ = roc_curve(patient_true_targets_thresh, patient_scores_thresh)
                        roc_auc_patient = auc(fpr_patient, tpr_patient)
                        ax_list_patient[i_thresh].plot(fpr_patient, tpr_patient)
                        #legend_labels_patient.append(key + ' (patient AUC=' + str(round(roc_auc_patient, 3)) + ')')
                        legend_labels_patient[i_thresh].append(key + ' (patient AUC=' + str(round(roc_auc_patient, 3)) + ')')

        test_n_patches = False
        if test_n_patches:
            n_list = np.arange(1, 100, 1)
            auc_n = np.zeros(n_list.shape)
            for ind, n in enumerate(n_list):
                auc_n[ind] = auc_for_n_patches(patch_scores, n, all_targets)
            plt.plot(n_list,auc_n)
            plt.xlabel('# of patches')
            plt.ylabel('test AUC score')
            plt.ylim([0,1])
            plt.xlim([1, 100])

    elif infer_type == 'MIL':
        roc_auc1, roc_auc_err, acc, acc_err, bacc, bacc_err, all_labels, all_targets, all_scores, total_pos, true_pos, total_neg, true_neg, num_slides = inference_data
        roc_auc.append(roc_auc1)

    EPS = 1e-7
    print(key)
    if not (multi_target) and (N_classes == 2):
        print('{} / {} correct classifications'.format(int(len(all_labels) - np.abs(np.array(all_targets) - np.array(all_labels)).sum()), len(all_labels)))
        balanced_acc = 100. * ((true_pos + EPS) / (total_pos + EPS) + (true_neg + EPS) / (total_neg + EPS)) / 2
        print('roc_auc:', roc_auc[-1])
        print('balanced_acc:', balanced_acc)
        print('np.sum(all_labels):', np.sum(all_labels))

    #temp RanS - calc BACC for each thresold
    plot_threshold = False
    if plot_threshold:
        bacc1 = np.zeros(20)
        tpr1 = np.zeros(20)
        tnr1 = np.zeros(20)
        threshs = np.arange(0, 1, 0.05)
        for ii, threshold in enumerate(threshs):
            all_preds = (all_scores > threshold).astype(int)
            true_pos1 = np.sum((all_preds == all_targets) & (all_preds == 1))
            true_neg1 = np.sum((all_preds == all_targets) & (all_preds == 0))
            bacc1[ii] = ((true_pos1 + EPS) / (total_pos + EPS) + (true_neg1 + EPS) / (total_neg + EPS)) / 2
            tpr1[ii] = true_pos1/(total_pos + EPS)
            tnr1[ii] = true_neg1 / (total_neg + EPS)
        plt.plot(threshs, tpr1,'r--')
        plt.plot(threshs, tnr1,'g-.')
        plt.plot(threshs, bacc1,'b-')
        plt.xlabel('score threshold')
        plt.legend(['tpr', 'tnr', 'BACC'],loc='lower left')

    calc_p_value = False
    if calc_p_value:
        n_iter = 10000
        rand_roc_auc = np.zeros(n_iter)
        N = len(all_labels)
        for ii in range(n_iter):
            #rand_preds = np.random.binomial(1, 0.79, size=[N, 1])
            rand_scores1 = np.random.permutation(all_scores)
            rand_roc_auc[ii] = roc_auc_score(all_targets, rand_scores1)
        p_value = np.sum(roc_auc1 <= rand_roc_auc)/n_iter

        #per patient
        n_iter = 10000
        rand_roc_auc = np.zeros(n_iter)
        N = len(patient_mean_score_df)
        for ii in range(n_iter):
            # rand_preds = np.random.binomial(1, 0.79, size=[N, 1])
            rand_scores1 = np.random.permutation(patient_mean_score_df['scores'])
            rand_roc_auc[ii] = roc_auc_score(patient_mean_score_df['targets'], rand_scores1)
        p_value_patient = np.sum(roc_auc_patient <= rand_roc_auc) / n_iter


#combine several models, RanS 11.4.21
combine_all_models = False
if patient_level and combine_all_models:
    slide_score_mean_all = np.mean(np.array(slide_score_all), axis=0)
    #patient_all = [all_slide_names[i][8:12] for i in range(all_slide_names.shape[0])]  # only TCGA!
    patch_all_df = pd.DataFrame({'patient': patient_all, 'scores': slide_score_mean_all, 'targets': all_targets})
    patient_mean_score_all_df = patch_all_df.groupby('patient').mean()
    roc_auc_all_patient = roc_auc_score(patient_mean_score_all_df['targets'], patient_mean_score_all_df['scores'])
    fpr_patient_all, tpr_patient_all, thresholds_patient_all = roc_curve(patient_mean_score_all_df['targets'],
                                                             patient_mean_score_all_df['scores'])
    plt.plot(fpr_patient_all, tpr_patient_all)
    legend_labels.append(' (all models combined patient AUC=' + str(round(roc_auc_all_patient, 3)) + ')')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.legend(legend_labels)

if N_classes > 2:
    for ax_i in range(N_classes - 1):
        box = ax_list[ax_i].get_position()
        ax_list[ax_i].set_position([box.x0, box.y0, box.width * 0.8, box.height])
        if patient_level:
            box = ax_list_patient[ax_i].get_position()
            ax_list_patient[ax_i].set_position([box.x0, box.y0, box.width * 0.8, box.height])
else:
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    if patient_level:
        box = ax1_patient.get_position()
        ax1_patient.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
if (N_classes == 2) or (multi_target):
    ax1.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(b=True)
    if patient_level:
        ax1_patient.legend(legend_labels_patient, loc='center left', bbox_to_anchor=(1, 0.5))
        ax1_patient.set_xlim(0, 1)
        ax1_patient.set_ylim(0, 1)
        ax1_patient.grid(b=True)
else:
    for i_thresh in range(N_classes-1):
        ax_list[i_thresh].legend(legend_labels[i_thresh], loc='center left', bbox_to_anchor=(1, 0.5))
        ax_list[i_thresh].set_xlim(0, 1)
        ax_list[i_thresh].set_ylim(0, 1)
        ax_list[i_thresh].grid(b=True)
        if patient_level:
            ax_list_patient[i_thresh].legend(legend_labels_patient[i_thresh], loc='center left', bbox_to_anchor=(1, 0.5))
            ax_list_patient[i_thresh].set_xlim(0, 1)
            ax_list_patient[i_thresh].set_ylim(0, 1)
            ax_list_patient[i_thresh].grid(b=True)

if N_classes == 2 or multi_target:
    fig1.savefig(os.path.join(inference_dir, inference_name + '_inference.png'), bbox_inches="tight")
    if patient_level:
        fig1_patient.savefig(os.path.join(inference_dir, inference_name + '_inference_patient.png'), bbox_inches="tight")
else:
    for i_thresh in range(N_classes - 1):
        fig_list[i_thresh].savefig(os.path.join(inference_dir, inference_name + '_thresh' + str(i_thresh) + '_inference.png'), bbox_inches="tight")
        if patient_level:
            fig_list_patient[i_thresh].savefig(os.path.join(inference_dir, inference_name + '_thresh' + str(i_thresh) + '_inference_patient.png'),bbox_inches="tight")

print('finished')
#plt.show()

