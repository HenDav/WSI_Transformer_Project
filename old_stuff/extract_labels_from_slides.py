import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pylibdmtx.pylibdmtx import decode as decode
import matplotlib.patches as patches
import os
import glob
import pandas as pd
import argparse
import openslide


parser = argparse.ArgumentParser(description='extract_labels_from_slides')
#parser.add_argument('--datafolder', type=str, default=r'/mnt/gipnetapp_public/sgils/2020-10-18', help='DataSet to use')
#parser.add_argument('--datafolder', type=str, default=r'/mnt/gipnetapp_public/sgils/BCF scans/Carmel Slides', help='DataSet to use')
parser.add_argument('--datafolder', type=str, default=r'C:\ran_data\Carmel_Slides_examples', help='DataSet to use')

args = parser.parse_args()

slide_dir = args.datafolder

slide_files = glob.glob(os.path.join(slide_dir, '*.mrxs'))
df = pd.DataFrame(columns=['data', 'file'])
for slide_file in slide_files:
    print('processing slide: ' + slide_file)
    try:
        img = openslide.open_slide(slide_file)
        #plt.imshow(img.associated_images['label'])
        code = decode(np.array(img.associated_images['label']))
        print('Found ' + str(len(code)) + ' barcodes')
        if len(code) == 1:
            barcode = code[0].data.decode('UTF-8')
            barcode_adjust = str(int(barcode[0:4]) - 9788) + '-' + barcode[4:]
        else:
            barcode_adjust = '-1'

        code_dict = {'data': barcode_adjust, 'file': os.path.basename(slide_file)}
        df = df.append(code_dict, ignore_index=True)
    except:
        pass
df.to_csv(os.path.join('slide_barcode_list.csv'))
print('Finished')