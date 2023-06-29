import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pylibdmtx.pylibdmtx import decode as decode
import matplotlib.patches as patches
import os
import glob
import pandas as pd
import time


if __name__ == '__main__':
    start_time = time.time()
    prev_time = start_time

    img_dir = r'C:\ran_data\RAMBAM\SlideID_images9'
    size_factor = 1 # try different values, 2 works better but takes much longer

    if not os.path.isdir(os.path.join(img_dir, 'out')):
        os.mkdir(os.path.join(img_dir, 'out'))

    img_files = glob.glob(os.path.join(img_dir, '*.jpg'))
    df = pd.DataFrame(columns=['barcode', 'left', 'top', 'h', 'w', 'file'])
    for img_file in img_files:
        print('processing image: ' + img_file)
        image = cv2.imread(img_file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_pil = Image.fromarray(gray)
        w, h = gray.shape
        gray_pil2 = gray_pil.resize((h * size_factor, w * size_factor), resample=Image.BICUBIC)
        all_codes = decode(np.array(gray_pil2))
        print('found ' + str(len(all_codes)) + ' barcodes')
        h_img = image.shape[0]

        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(image, aspect='auto')
        for code in all_codes:
            left = code.rect.left/size_factor
            top = code.rect.top/size_factor
            height = code.rect.height/size_factor
            width = code.rect.width/size_factor
            barcode = code.data.decode('UTF-8')
            if  15 >= len(code.data) >= 14: #barcodes are 14 or 15 characters long
                barcode_adjust = str(int(barcode[0:4])-9788) + '-' + barcode[4:]
                code_dict = {'barcode': barcode_adjust, 'left': int(left), 'top': int(top), 'h': int(height), 'w': int(width), 'file': os.path.basename(img_file)}
                df = df.append(code_dict, ignore_index=True)
                rect = patches.Rectangle((left, h_img - top - height), width, height, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        fig.savefig(os.path.join(img_dir, 'out', os.path.splitext(os.path.basename(img_file))[0] + '_results.jpg'), dpi=300)
        plt.close(fig)
        df.to_csv(os.path.join(img_dir, 'out', 'barcode_list.csv')) #save after every image
        image_time = time.time()
        print('processing time: ' + str(image_time - prev_time) + ' sec')
        prev_time = image_time

    print('Finished, total time: ' + str(time.time() - start_time) + ' sec')

