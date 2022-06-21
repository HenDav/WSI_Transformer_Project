from Dataset_Maker import dataset_maker
Dataset = 'Her2_1'
data_dir = r'C:\ran_data\Carmel_Slides_examples\Her2\Batch_1\slides'
#Dataset = 'CARMEL9'
#data_dir = r'C:\ran_data\Carmel_Slides_examples\CARMEL9'
scan_barcodes = False
get_slide_labels = True
step = 0

dataset_maker.prepare_dataset_for_training(Dataset=Dataset,
                                           data_dir=data_dir,
                                           get_slide_labels=get_slide_labels,
                                           scan_barcodes=scan_barcodes,
                                           step=step)
