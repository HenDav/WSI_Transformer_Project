import pandas as pd
import os
from Dataset_Maker import dataset_utils
import numpy as np


def remove_slides_according_to_list(in_dir, dataset):
    #TODO arrange
    if dataset == 'CARMEL':
        batches = np.arange(1, 9, 1)
        ignore_list_file = r'C:\ran_data\Carmel_Slides_examples\folds_labels\Slides_to_discard.xlsx'
        ignore_list_DF = pd.read_excel(ignore_list_file)
        ignore_list = list(ignore_list_DF['Slides to discard'])

        ignore_list_file_blur = r'C:\ran_data\Carmel_Slides_examples\folds_labels\blurry_slides_to_discard.xlsx'
        ignore_list_DF_blur = pd.read_excel(ignore_list_file_blur)
        ignore_list_blur = list(ignore_list_DF_blur['Slides to discard'])

        labels_data_file = os.path.join(in_dir, 'Carmel_annotations_25-10-2021.xlsx')

    elif dataset == 'HAEMEK':
        batches = np.arange(1, 2, 1)
        ignore_list = []
        ignore_list_blur = []
        labels_data_file = os.path.join(in_dir, 'Afula_annotations_13-01-22.xlsx')

    elif dataset == 'BENIGN':
        batches = np.arange(1, 4, 1)
        ignore_list_file = r'C:\ran_data\Benign\Slides_to_discard.xlsx'
        ignore_list_DF = pd.read_excel(ignore_list_file)
        ignore_list = list(ignore_list_DF['Slides to discard'])
        ignore_list_blur = []
        labels_data_file = os.path.join(in_dir, 'Carmel_annotations_Benign_merged_09-01_22.xlsx')


    label_data_DF = pd.read_excel(labels_data_file)
    data_field = label_data_DF.keys().to_list() #take all fields
    #binary_data_fields = ['ER status', 'PR status', 'Her2 status', 'Ki67 status'] #fields which should be translated from 0,1 to Negative, Positive
    binary_data_fields = [] #fields which should be translated from 0,1 to Negative, Positive

    for batch in batches:
        print('batch ', str(batch))
        if dataset == 'CARMEL':
            meta_data_file = os.path.join(in_dir, 'slides_data_Carmel' + str(batch) + '.xlsx')
        else:
            meta_data_file = os.path.join(in_dir, 'slides_data_' + dataset + str(batch) + '.xlsx')

        meta_data_DF = pd.read_excel(meta_data_file)
        meta_data_DF['slide barcode'] = [fn[:-5] for fn in meta_data_DF['file']]
        #for ind, slide in enumerate(meta_data_DF['patient barcode']):
        for ind, slide in enumerate(meta_data_DF['slide barcode']):
            #skip files in the ignore list and the blur_ignore list
            if (slide.replace('_', '/') in ignore_list) or (slide.replace('_', '/') in ignore_list_blur):
                for field in data_field:
                    #meta_data_DF.loc[meta_data_DF['patient barcode'] == slide, field] = 'Missing Data'
                    meta_data_DF.loc[meta_data_DF['slide barcode'] == slide, field] = 'Missing'
                continue

            slide_tissue_id = slide.split('_')[0] + '/' + slide.split('_')[1]
            slide_block_id = slide.split('_')[2]
            slide_label_data = label_data_DF.loc[label_data_DF['TissueID'] == slide_tissue_id]

            if len(slide_label_data) == 0:
                #no matching tissue id
                print('1. Batch ' + str(batch) + ': could not find tissue_id ' + str(slide_tissue_id) + ' in annotations file, for slide ' + slide)

            elif len(slide_label_data) == 1:
                #one matching tissue id, make sure block id is empty or matching
                BlockID = slide_label_data['BlockID'].item()
                if np.isnan(BlockID) or str(BlockID) == slide_block_id:
                    for field in data_field:
                        #meta_data_DF.loc[meta_data_DF['patient barcode'] == slide, field] = slide_label_data[field].values[0]
                        meta_data_DF.loc[meta_data_DF['slide barcode'] == slide, field] = slide_label_data[field].values[0]
                else:
                    print('2. Batch ' + str(batch) + ': one matching tissue_id for ', str(slide_tissue_id), ', could not find matching blockID ' + str(slide_block_id) + ' in annotations file, for slide ' + slide)

            elif len(slide_label_data) > 1:
                slide_label_data_block = slide_label_data[slide_label_data['BlockID'] == int(slide_block_id)]
                if len(slide_label_data_block) == 0:
                    print('3: Batch ' + str(batch) + ': ', str(len(slide_label_data)), ' matching tissue_id for ',
                          str(slide_tissue_id), ', could not find matching blockID ' + slide_block_id + ' in annotations file, for slide ' + slide)
                elif len(slide_label_data_block) > 1:
                    print('4: Batch ' + str(batch) + ': ', str(len(slide_label_data)), ' matching tissue_id for ',
                          str(slide_tissue_id), ', found more than one matching blockID ' + slide_block_id + ' in annotations file, for slide ' + slide)
                else:
                    for field in data_field:
                        #meta_data_DF.loc[meta_data_DF['patient barcode'] == slide, field] = slide_label_data[field].values[0]
                        #meta_data_DF.loc[meta_data_DF['patient barcode'] == slide, field] = slide_label_data_block[field].values[0]
                        meta_data_DF.loc[meta_data_DF['slide barcode'] == slide, field] = slide_label_data_block[field].values[0]

        #replace empty cells with "missing data", and 0,1 with "Positive", "Negative"
        #for field in data_field:
        for field in data_field:
            meta_data_DF[field] = meta_data_DF[field].replace(np.nan, 'Missing Data', regex=True)
            meta_data_DF[field] = meta_data_DF[field].replace('Missing', 'Missing Data', regex=True)

        for field in binary_data_fields:
            meta_data_DF[field] = meta_data_DF[field].replace(1, 'Positive', regex=True)
            meta_data_DF[field] = meta_data_DF[field].replace(0, 'Negative', regex=True)

        dataset_utils.save_df_to_excel(meta_data_DF, os.path.join(in_dir, 'slides_data_' + dataset + str(batch) + '_labeled.xlsx'))
