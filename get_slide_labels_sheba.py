import os
import numpy as np
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='get_slide_labels_sheba')
#parser.add_argument('--in_dir', default=r'C:\ran_data\Carmel_Slides_examples\folds_labels', type=str, help='input dir')
parser.add_argument('--in_dir', default=r'C:\ran_data\Sheba', type=str, help='input dir')
args = parser.parse_args()

in_dir = args.in_dir
batches = np.arange(1, 5, 1)

labels_data_file = os.path.join(in_dir, 'SHEBA ONCOTYPE 160122_Ran.xlsx')
label_data_DF = pd.read_excel(labels_data_file)
#data_field = ['ER status', 'PR status', 'Her2 status', 'TissueType', 'PatientIndex']
#data_field = ['ER status', 'PR status', 'Her2 status', 'TissueType', 'PatientIndex', 'Ki67 status', 'ER score',
#              'PR score', 'Her2 score', 'Ki67 score', 'Age', 'Grade']
#binary_data_fields = ['ER status', 'PR status', 'Her2 status', 'Ki67 status']
#data_field = ['ER100 status'] #RanS 14.11.21
data_field = label_data_DF.keys().to_list() #take all fields, RanS 16.1.22

for batch in batches:
    print('batch ', str(batch))
    meta_data_file = os.path.join(in_dir, 'slides_data_SHEBA_batch' + str(batch) + '.xlsx')
    meta_data_DF = pd.read_excel(meta_data_file)
    #meta_data_DF['slide barcode'] = [fn[:-5] for fn in meta_data_DF['file']]
    #for ind, slide in enumerate(meta_data_DF['patient barcode']):
    for ind, slide in enumerate(meta_data_DF['patient barcode']):
        #slide_tissue_id = slide.split('_')[0] + '/' + slide.split('_')[1]
        #slide_block_id = slide.split('_')[2]
        slide_label_data = label_data_DF.loc[label_data_DF['Code'] == slide]

        if len(slide_label_data) == 0:
            #no matching tissue id
            print('1. Batch ' + str(batch) + ': could not find data in annotations file, for slide ' + slide)

        elif len(slide_label_data) == 1:
            #one matching tissue id, make sure block id is empty or matching
            #BlockID = slide_label_data['BlockID'].item()
            #if np.isnan(BlockID) or str(BlockID) == slide_block_id:
            for field in data_field:
                    #meta_data_DF.loc[meta_data_DF['patient barcode'] == slide, field] = slide_label_data[field].values[0]
                meta_data_DF.loc[meta_data_DF['patient barcode'] == slide, field] = slide_label_data[field].values[0]
            onco_score = slide_label_data['Oncotype DX Breast Cancer Assay'].values[0]

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


        elif len(slide_label_data) > 1:
            print('2: Batch ' + str(batch) + ': ', str(len(slide_label_data)), 'found more than one match in annotations file, for slide ' + slide)

    #replace empty cells with "missing data", and 0,1 with "Positive", "Negative"
    #for field in data_field:
    for field in data_field:
        meta_data_DF[field] = meta_data_DF[field].replace(np.nan, 'Missing Data', regex=True)
        #meta_data_DF[field] = meta_data_DF[field].replace('Missing', 'Missing Data', regex=True)

    meta_data_DF.to_excel(os.path.join(in_dir, 'slides_data_SHEBA_batch' + str(batch) + '_labeled.xlsx'))

#create merged files
df_list = []
df_list2 = []

for batch in batches:
    meta_data_file = os.path.join(in_dir, 'slides_data_SHEBA_batch' + str(batch) + '.xlsx')
    meta_data_DF = pd.read_excel(meta_data_file)
    df_list.append(meta_data_DF)

    meta_data_file2 = os.path.join(in_dir, 'slides_data_SHEBA_batch' + str(batch) + '_labeled.xlsx')
    meta_data_DF2 = pd.read_excel(meta_data_file2)
    df_list2.append(meta_data_DF2)

merged_DF = pd.concat(df_list)
merged_DF2 = pd.concat(df_list2)

merged_DF.to_excel(os.path.join(in_dir, 'slides_data_SHEBA_merged.xlsx'))
merged_DF2.to_excel(os.path.join(in_dir, 'slides_data_SHEBA_labeled_merged.xlsx'))

print('finished')