import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pylibdmtx.pylibdmtx import decode as decode
import matplotlib.patches as patches
import os
import glob
import pandas as pd
#import pytesseract
#from pytesseract import Output
#from pyzbar.pyzbar import decode as decode_1d #for 1d barcodes, if needed
import time
import re
import argparse

parser = argparse.ArgumentParser(description='Barcode_detection')
parser.add_argument('-ds', '--img_dir', type=str, default=r'C:\ran_data\RAMBAM\AH1-2', help='input directory')
parser.add_argument('--is_haemek', action='store_true', help='data is not from Carmel')
args = parser.parse_args()

'''def censor_text(img_file):
    # RanS 12.11.20 - find and censor text.
    # doesn't really work

    # img_text = pytesseract.image_to_string(img_file)
    d = pytesseract.image_to_data(img_file, output_type=Output.DICT, lang='eng+heb')
    n_boxes = len(d['level'])
    image2 = image.copy()
    for ii in range(n_boxes):
        (x, y, w, h) = (d['left'][ii], d['top'][ii], d['width'][ii], d['height'][ii])
        # if d['conf'][ii] == '-1':
        #    cv2.rectangle(image2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # else:
        # if int(d['conf'][ii]) > 50:
        if any("\u0590" <= c <= "\u05EA" for c in d['text'][ii]):
            cv2.rectangle(image2, (x, y), (x + w, y + h), (255, 0, 0), 3)
    plt.imshow(image2)

    d1 = pytesseract.image_to_boxes(img_file, output_type=Output.DICT, lang='eng+heb')
    n_boxes = len(d1['char'])
    image2 = image.copy()
    imh, imw = image.shape[:2]
    for ii in range(n_boxes):
        # (x, y, w, h) = (d['left'][ii], d['top'][ii], d['width'][ii], d['height'][ii])
        (x1, y1, x2, y2) = (d1['left'][ii], d1['top'][ii], d1['right'][ii], d1['bottom'][ii])
        # if d['conf'][ii] == '-1':
        #    cv2.rectangle(image2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # else:
        # if int(d['conf'][ii]) > 50:
        # if any("\u0590" <= c <= "\u05EA" for c in d['text'][ii]):
        cv2.rectangle(image2, (x1, imh - y1), (x2, imh - y2), (255, 0, 0), 3)
    plt.imshow(image2)
    # cv2.imshow('image', image)
    # cv2.waitKey(0)'''

if __name__ == '__main__':
    start_time = time.time()
    prev_time = start_time
    #pytesseract.pytesseract.tesseract_cmd = r'C:\ran_programs\Tesseract-OCR\tesseract'

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
        image = cv2.imread(img_file)

        #RanS 12.11.20 - find and censor text
        #use_censor_text = False
        #if use_censor_text:
        #    image = censor_text(image)

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

        #merge both results
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
            try: #sometimes it finds garbage codes
                barcode = code.data.decode('UTF-8')
                if barcode in unique_barcodes:
                    continue
                else:
                    unique_barcodes.append(barcode)
                if is_carmel:
                    barcode_adjust = str(int(barcode[0:4])-9788) + '-' + barcode[4:]
                else: #haemek
                    year = int(barcode[4])+14
                    barcode_adjust = str(year) + '-' + barcode[5:]
                if re.search("\d{2}-\d+/\d+/\d+/\w", barcode_adjust) != None or is_carmel is False:
                    code_dict = {'SlideID': barcode_adjust, 'file': os.path.basename(img_file)}
                    df = df.append(code_dict, ignore_index=True)
                    rect = patches.Rectangle((left, h_img - top - height), width, height, linewidth=2, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                    num_valids += 1
            except:
                pass

        print('found ' + str(len(unique_barcodes)) + ' barcodes, of which ', str(num_valids), ' are valid')
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        fig.savefig(os.path.join(img_dir, 'out', os.path.splitext(os.path.basename(img_file))[0] + '_results.jpg'), dpi=300)
        plt.close(fig)
        df.to_excel(os.path.join(img_dir, 'out', 'barcode_list.xlsx'))  #save after every image
        image_time = time.time()
        print('processing time: ' + str(image_time - prev_time) + ' sec')
        prev_time = image_time

    print('Finished, total time: ' + str(time.time() - start_time) + ' sec')