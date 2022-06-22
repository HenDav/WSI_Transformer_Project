from Segmentation.remove_control_tissue import remove_control_tissue_rows_from_segmentation
import os
from PIL import Image
import numpy as np

dirname = r'C:\Users\User\Technion\Karin Stoliar - thumb'
fname = 'Inked0034_0_thumb_21-1733_1_7_d_LI.jpg'
img_file = os.path.join(dirname, fname)
img = np.array(Image.open(img_file))

img_clear = remove_control_tissue_rows_from_segmentation(img)