from Dataset_Maker import dataset_maker
Dataset = 'Her2_1'
data_dir = r'C:\ran_data\Carmel_Slides_examples\Her2\Batch_1'
save_label_image = True
has_barcodes = True

dataset_maker.prepare_dataset_for_training(Dataset=Dataset, data_dir=data_dir, save_label_image=save_label_image, has_barcodes=has_barcodes)
