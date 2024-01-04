import os
import numpy as np
import pandas as pd


def get_onco_score_binary_status(meta_data_DF, onco_score, slide):
    if onco_score < 11:
        meta_data_DF.loc[meta_data_DF['patient barcode'] == slide, 'onco_score_11 status'] = 'Negative'
        meta_data_DF.loc[meta_data_DF['patient barcode'] == slide, 'onco_score_18 status'] = 'Negative'
        meta_data_DF.loc[meta_data_DF['patient barcode'] == slide, 'onco_score_26 status'] = 'Negative'
        meta_data_DF.loc[meta_data_DF['patient barcode'] == slide, 'onco_score_31 status'] = 'Negative'
        meta_data_DF.loc[meta_data_DF['patient barcode'] == slide, 'onco_score_all status'] = '0'
    elif (onco_score >= 11) and (onco_score < 18):
        meta_data_DF.loc[meta_data_DF['patient barcode'] == slide, 'onco_score_11 status'] = 'Positive'
        meta_data_DF.loc[meta_data_DF['patient barcode'] == slide, 'onco_score_18 status'] = 'Negative'
        meta_data_DF.loc[meta_data_DF['patient barcode'] == slide, 'onco_score_26 status'] = 'Negative'
        meta_data_DF.loc[meta_data_DF['patient barcode'] == slide, 'onco_score_31 status'] = 'Negative'
        meta_data_DF.loc[meta_data_DF['patient barcode'] == slide, 'onco_score_all status'] = '1'
    elif (onco_score >= 18) and (onco_score < 26):
        meta_data_DF.loc[meta_data_DF['patient barcode'] == slide, 'onco_score_11 status'] = 'Positive'
        meta_data_DF.loc[meta_data_DF['patient barcode'] == slide, 'onco_score_18 status'] = 'Positive'
        meta_data_DF.loc[meta_data_DF['patient barcode'] == slide, 'onco_score_26 status'] = 'Negative'
        meta_data_DF.loc[meta_data_DF['patient barcode'] == slide, 'onco_score_31 status'] = 'Negative'
        meta_data_DF.loc[meta_data_DF['patient barcode'] == slide, 'onco_score_all status'] = '2'
    elif (onco_score >= 26) and (onco_score < 31):
        meta_data_DF.loc[meta_data_DF['patient barcode'] == slide, 'onco_score_11 status'] = 'Positive'
        meta_data_DF.loc[meta_data_DF['patient barcode'] == slide, 'onco_score_18 status'] = 'Positive'
        meta_data_DF.loc[meta_data_DF['patient barcode'] == slide, 'onco_score_26 status'] = 'Positive'
        meta_data_DF.loc[meta_data_DF['patient barcode'] == slide, 'onco_score_31 status'] = 'Negative'
        meta_data_DF.loc[meta_data_DF['patient barcode'] == slide, 'onco_score_all status'] = '3'
    elif (onco_score >= 31):
        meta_data_DF.loc[meta_data_DF['patient barcode'] == slide, 'onco_score_11 status'] = 'Positive'
        meta_data_DF.loc[meta_data_DF['patient barcode'] == slide, 'onco_score_18 status'] = 'Positive'
        meta_data_DF.loc[meta_data_DF['patient barcode'] == slide, 'onco_score_26 status'] = 'Positive'
        meta_data_DF.loc[meta_data_DF['patient barcode'] == slide, 'onco_score_31 status'] = 'Positive'
        meta_data_DF.loc[meta_data_DF['patient barcode'] == slide, 'onco_score_all status'] = '4'
    else:
        meta_data_DF.loc[meta_data_DF['patient barcode'] == slide, 'onco_score_11 status'] = 'Missing Data'
        meta_data_DF.loc[meta_data_DF['patient barcode'] == slide, 'onco_score_18 status'] = 'Missing Data'
        meta_data_DF.loc[meta_data_DF['patient barcode'] == slide, 'onco_score_26 status'] = 'Missing Data'
        meta_data_DF.loc[meta_data_DF['patient barcode'] == slide, 'onco_score_31 status'] = 'Missing Data'
        meta_data_DF.loc[meta_data_DF['patient barcode'] == slide, 'onco_score_all status'] = 'Missing Data'
    return meta_data_DF


def get_slide_labels_sheba(in_dir, batches, label_file):

    label_data_DF = pd.read_excel(os.path.join(in_dir, label_file))
    data_field = label_data_DF.keys().to_list()  # take all fields

    for batch in batches:
        print('batch ', str(batch))
        meta_data_file = os.path.join(in_dir, 'slides_data_SHEBA' + str(batch) + '.xlsx')
        meta_data_DF = pd.read_excel(meta_data_file)
        for ind, slide in enumerate(meta_data_DF['patient barcode']):
            slide_label_data = label_data_DF.loc[label_data_DF['CodeID'] == slide]

            if len(slide_label_data) == 0:
                #no matching tissue id
                print('1. Batch ' + str(batch) + ': could not find data in annotations file, for slide ' + slide)

            elif len(slide_label_data) == 1:
                #one matching tissue id, make sure block id is empty or matching
                for field in data_field:
                    meta_data_DF.loc[meta_data_DF['patient barcode'] == slide, field] = slide_label_data[field].values[0]
                onco_score = slide_label_data['Oncotype DX Breast Cancer Assay'].values[0]

                meta_data_DF = get_onco_score_binary_status(meta_data_DF, onco_score, slide)

            elif len(slide_label_data) > 1:
                print('2: Batch ' + str(batch) + ': ', str(len(slide_label_data)), 'found more than one match in annotations file, for slide ' + slide)

        #replace empty cells with "missing data", and 0,1 with "Positive", "Negative"
        for field in data_field:
            meta_data_DF[field] = meta_data_DF[field].replace(np.nan, 'Missing Data', regex=True)

        meta_data_DF.to_excel(os.path.join(in_dir, 'slides_data_SHEBA' + str(batch) + '_labeled.xlsx'))


def create_merged_metadata_file_sheba(in_dir, batches):
    df_list = []

    for batch in batches:
        meta_data_file = os.path.join(in_dir, 'SHEBA' + str(batch), 'slides_data_SHEBA' + str(batch) + '.xlsx')
        meta_data_DF = pd.read_excel(meta_data_file)
        df_list.append(meta_data_DF)

    merged_DF = pd.concat(df_list)

    merged_DF.to_excel(os.path.join(in_dir, 'slides_data_SHEBA_merged.xlsx'))