import subprocess

subprocess.run(['python', 'prepare_data.py',
                    #'--segmentation',
                    #'--data_collection',
                    '--grid',
                    #'--stats',
                    #'--data_folder', r'C:\ran_data\TCGA',
                    #'--data_root', r'C:\ran_data\TCGA_example_slides\TCGA_examples_131020_flat',
                    #'--data_root', r'C:\ran_data\TCGA_example_slides\bad_seg_230221',
                    '--data_root', r'C:\ran_data\ABCTB\ABCTB_examples',
                    #'--data_folder', r'C:\ran_data\TCGA_example_slides\TCGA_bad_examples_181020',
                    #'--data_folder', r'C:\ran_data\gip-main_examples\Lung',
                    #'--data_root', r'C:\ran_data\Lung_examples',
                    #'--data_root', r'C:\ran_data\herohe_grid_debug_temp_060121',
                    #'--data_root', r'C:\ran_data\RedSquares',
                    #'--data_root', r'C:\ran_data\Carmel_Slides_examples',
                    #'--data_root', r'C:\ran_data\HEROHE_examples',

                    #'--dataset','HEROHE',
                    #'--dataset','CARMEL3',
                    #'--dataset', 'TCGA',
                    '--dataset', 'ABCTB',
                    '--tissue_coverage', '0.3',
                    '--mag', '10',
                    '--multi'
                    #'--data_folder', r'C:\ran_data\gip-main_examples\Leukemia',
                    #'--data_folder', r'C:\ran_data\ABCTB',
                ])