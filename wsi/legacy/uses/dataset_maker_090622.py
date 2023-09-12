import dataset_maker

#Dataset = 'Her2_1'
#data_dir = r'C:\ran_data\Carmel_Slides_examples\Her2\Batch_1\slides'
Dataset = 'CARMEL9'
data_dir = r'C:\ran_data\Carmel_Slides_examples\CARMEL9'
scan_barcodes = True
get_slide_labels = True
step = 1

tile_size = 256
mag = 10
tissue_coverage = 0.3
control_tissue = True

dataset_maker.prepare_dataset_for_training(Dataset=Dataset,
                                           data_dir=data_dir,
                                           get_slide_labels=get_slide_labels,
                                           scan_barcodes=scan_barcodes,
                                           step=step,
                                           tile_size=tile_size,
                                           mag=mag,
                                           tissue_coverage=tissue_coverage,
                                           control_tissue=control_tissue)
