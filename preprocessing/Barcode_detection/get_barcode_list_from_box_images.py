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
import re
import argparse

parser = argparse.ArgumentParser(description='Barcode_detection')
parser.add_argument('-ds', '--img_dir', type=str, default=r'C:\ran_data\BOX_IMAGES\CO1-2_taken\temp', help='input directory')
parser.add_argument('--is_haemek', action='store_true', help='data is not from Carmel')
args = parser.parse_args()


def get_box_name(fn):
    img_file_basename = os.path.basename(fn)
    split_name = img_file_basename.split('_')
    if len(split_name) == 2:
        return split_name[0]
    else:
        raise IOError('Image File name should be in the format "BOXNUM_NUM", i.e. "H21_1.jpeg"')


if __name__ == '__main__':
    start_time = time.time()
    prev_time = start_time

    img_dir = args.img_dir
    is_carmel = not args.is_haemek
    size_factor = 1

    if not os.path.isdir(os.path.join(img_dir, 'out')):
        os.mkdir(os.path.join(img_dir, 'out'))

    img_files_jpeg = glob.glob(os.path.join(img_dir, '*.jpeg'))
    img_files_jpg = glob.glob(os.path.join(img_dir, '*.jpg'))
    img_files = img_files_jpeg + img_files_jpg
    df = pd.DataFrame(columns=['file', 'index', 'Box', 'SlideID', 'Comments'])
    for img_file in img_files:
        print('processing image: ' + img_file)
        box = get_box_name(img_file)
        image = cv2.imread(img_file)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_pil = Image.fromarray(gray)
        w, h = gray.shape
        if size_factor != 1:
            gray_pil2 = gray_pil.resize((h * size_factor, w * size_factor), resample=Image.BICUBIC)
        else:
            gray_pil2 = gray_pil

        all_codes1 = decode(np.array(gray_pil2))
        _, seg_map = cv2.threshold(np.array(gray_pil2), 0, 255, cv2.THRESH_OTSU)
        all_codes2 = decode(np.array(seg_map))

        # merge both results
        all_codes = all_codes1 + all_codes2
        unique_barcodes = []
        h_img = image.shape[0]

        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(image, aspect='auto')
        num_valids = 0
        for code in all_codes:
            left = code.rect.left/size_factor
            top = code.rect.top/size_factor
            height = code.rect.height/size_factor
            width = code.rect.width/size_factor
            try:  # sometimes it finds garbage codes
                barcode = code.data.decode('UTF-8')
                if barcode in unique_barcodes:
                    continue
                else:
                    unique_barcodes.append(barcode)
                if is_carmel:
                    if int(barcode[:2]) < 90:  # new format, no change required
                        barcode_adjust = barcode
                    else:
                        year = int(barcode[0:4]) - 9788
                        if year > 200:  # year 21 shows as 201, and so on
                            year -= 180
                        barcode_adjust = str(year) + '-' + barcode[4:]
                else:  # haemek
                    year = int(barcode[4])+14
                    barcode_adjust = str(year) + '-' + barcode[5:]
                if re.search("\d{2}-\d+/\d+/\d+/\w", barcode_adjust) is not None or is_carmel is False:
                    code_dict = {'SlideID': barcode_adjust, 'file': os.path.basename(img_file), 'Box': box}
                    df = df.append(code_dict, ignore_index=True)
                    rect = patches.Rectangle((left, h_img - top - height), width, height,
                                             linewidth=2, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                    num_valids += 1
            except:
                pass

        print('found ' + str(len(unique_barcodes)) + ' barcodes, of which ', str(num_valids), ' are valid')
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        fig.savefig(os.path.join(img_dir, 'out', os.path.splitext(os.path.basename(img_file))[0] + '_results.jpg'), dpi=300)
        plt.close(fig)
        df.to_excel(os.path.join(img_dir, 'out', 'barcode_list.xlsx'))  # save after every image
        image_time = time.time()
        print('processing time: ' + str(image_time - prev_time) + ' sec')
        prev_time = image_time

    print('Finished, total time: ' + str(time.time() - start_time) + ' sec')
