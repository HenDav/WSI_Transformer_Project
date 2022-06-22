from Dataset_Maker import slide_walker as sw

def prepare_dataset_for_training(Dataset, data_dir, save_label_image, has_barcodes):
    sw.create_slide_list(data_dir, Dataset)

    sw.add_barcodes_to_slide_list(data_dir, Dataset, save_label_image, has_barcodes)

