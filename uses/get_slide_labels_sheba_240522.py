from get_slide_labels_sheba import get_slide_labels_sheba
import numpy as np

in_dir = r'C:\ran_data\Sheba'
labels_file = r'Sheba_Oncotype_2015-2020_09-05-22.xlsx'
#batches = np.arange(1, 5, 1)
batches = np.arange(5, 7, 1)

get_slide_labels_sheba(in_dir, batches, labels_file)