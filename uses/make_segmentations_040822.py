import utils_data_managment

Dataset = 'HER2_1'
data_dir = r'C:\ran_data\Carmel_Slides_examples\Her2\Batch_1'

utils_data_managment.make_segmentations(DataSet=Dataset,
                                        ROOT_DIR=data_dir,
                                        out_path=data_dir,
                                        num_workers=1)

utils_data_managment.make_grid(DataSet=Dataset,
                                ROOT_DIR=data_dir,
                                tile_sz=256,
                                tissue_coverage=0.3,
                                desired_magnification=10,
                                num_workers=1)