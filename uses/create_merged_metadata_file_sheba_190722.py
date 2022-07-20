from Dataset_Maker.get_slide_labels_sheba import create_merged_metadata_file_sheba

indir = r'C:\ran_data\Sheba'
batches = range(1, 7)

create_merged_metadata_file_sheba(indir, batches)