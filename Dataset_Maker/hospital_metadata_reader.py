import os
import numpy as np
import argparse
import pandas as pd
from Dataset_Maker import dataset_utils
from pathlib import Path


def add_hospital_labels_to_metadata(in_dir, dataset, hospital_metadata_file):
    # TODO split to subfunctions
    labels_data_file = os.path.join(in_dir, hospital_metadata_file)
    label_data_DF = pd.read_excel(labels_data_file)
    assert ('slide barcode' in label_data_DF.columns), "column 'slide barcode' missing in hospital metadata file"
    data_field = label_data_DF.keys().to_list()  # take all fields

    dataset_group = dataset_utils.get_dataset_group(dataset)
    if dataset_group.name in ['CARMEL', 'HAEMEK', 'BENIGN', 'HER2']:
        match_by_tissue_id = True
        assert ('TissueID' in label_data_DF.columns), "column 'TissueID' missing in hospital metadata file"
        assert ('BlockID' in label_data_DF.columns), "column 'BlockID' missing in hospital metadata file"
    else:
        match_by_tissue_id = False
    assert ('PatientID' in label_data_DF.columns), "column 'PatientID' missing in hospital metadata file"

    slides_data_file = dataset_utils.get_slides_data_file(in_dir, dataset)
    dataset_utils.backup_dataset_metadata(slides_data_file, '_before_hospital_metadata')
    meta_data_DF = dataset_utils.open_excel_file(slides_data_file)

    meta_data_DF['slide barcode'] = [Path(fn).stem for fn in meta_data_DF['file']]

    for ind, slide in enumerate(meta_data_DF['slide barcode']):
        if match_by_tissue_id:
            slide_tissue_id = slide.split('_')[0] + '/' + slide.split('_')[1]
            slide_block_id = slide.split('_')[2]
            slide_label_data = label_data_DF.loc[label_data_DF['TissueID'] == slide_tissue_id]
        else:
            slide_label_data = label_data_DF.loc[label_data_DF['slide barcode'] == slide]

        found_one_match = False
        if len(slide_label_data) == 0:
            # no matching tissue id
            print('Could not find match in annotations file, for slide ' + slide)
        elif len(slide_label_data) == 1:
            if match_by_tissue_id:
                # one matching tissue id, make sure block id is empty or matching
                BlockID = slide_label_data['BlockID'].item()
                if np.isnan(BlockID) or str(BlockID) == slide_block_id:
                    found_one_match = True
                else:
                    print('One matching tissue_id for ', str(slide_tissue_id),
                          ', could not find matching blockID ', str(slide_block_id),
                          ' in annotations file, for slide ' + slide)
            else:
                found_one_match = True

            if found_one_match:
                for field in data_field:
                    meta_data_DF.loc[meta_data_DF['slide barcode'] == slide, field] = slide_label_data[field].values[0]
                meta_data_DF.loc[meta_data_DF['slide barcode'] == slide, 'patient barcode'] = slide_label_data['PatientID'].values[0]

        elif len(slide_label_data) > 1:
            if match_by_tissue_id:
                slide_label_data_block = slide_label_data[slide_label_data['BlockID'] == int(slide_block_id)]
                if len(slide_label_data_block) == 0:
                    print(str(len(slide_label_data)), ' matching tissue_id for ', str(slide_tissue_id),
                          ', could not find matching blockID ' + slide_block_id + ' in annotations file, for slide ' + slide)
                elif len(slide_label_data_block) > 1:
                    print(str(len(slide_label_data)), ' matching tissue_id for ', str(slide_tissue_id),
                          ', found more than one matching blockID ' + slide_block_id + ' in annotations file, for slide ' + slide)
                else:
                    for field in data_field:
                        meta_data_DF.loc[meta_data_DF['slide barcode'] == slide, field] = \
                        slide_label_data_block[field].values[0]
                    meta_data_DF.loc[meta_data_DF['slide barcode'] == slide, 'patient barcode'] = \
                    slide_label_data_block['PatientID'].values[0]
            else:
                print(str(len(slide_label_data)),
                      'found more than one match in annotations file, for slide ' + slide)

    for field in data_field:
        meta_data_DF[field] = meta_data_DF[field].replace('Missing', 'Missing Data', regex=True)
        meta_data_DF[field] = meta_data_DF[field].replace(np.nan, 'Missing Data', regex=True)

    dataset_utils.save_df_to_excel(meta_data_DF, slides_data_file)


def binarize_labels(in_dir, dataset, binary_data_fields):
    # fields which should be translated from 0,1 to Negative, Positive
    # binary_data_fields = ['ER status', 'PR status', 'Her2 status', 'Ki67 status']
    backup_ext = '_before_data_fields_binarization'

    slides_data_file, slides_data_DF = dataset_utils.load_backup_slides_data(in_dir, dataset, extension=backup_ext)

    # SAVE RAW DATA
    #binary_data_fields = []
    for field in binary_data_fields:
        slides_data_DF[field] = slides_data_DF[field].replace(1, 'Positive', regex=True)
        slides_data_DF[field] = slides_data_DF[field].replace(0, 'Negative', regex=True)

    dataset_utils.save_df_to_excel(slides_data_DF, slides_data_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='hospital metadata reader')
    parser.add_argument('--in_dir', default=r'C:\ran_data\Benign', type=str, help='input dir')
    parser.add_argument('--dataset', default=r'BENIGN', type=str, help='name of dataset')  # CARMEL, HAEMEK, BENIGN
    args = parser.parse_args()

    in_dir = args.in_dir

    pass  # TODO
