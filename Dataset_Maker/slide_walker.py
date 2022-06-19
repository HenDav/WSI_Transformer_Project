import os
import openslide
import matplotlib.pyplot as plt
from pylibdmtx.pylibdmtx import decode as decode
import pandas as pd
import time
from PIL import Image
import re
from collections import OrderedDict
from tqdm import tqdm
import numpy as np

BARCODE_CONVENTION_NONE = 0
BARCODE_CONVENTION_CARMEL = 1
BARCODE_CONVENTION_HAEMEK = 2

EMPTY_BARCODE_LIST = []


def create_slide_list(walk_dir, dataset_name):
    print('creating slide list for data directory = ' + walk_dir)

    excel_file = get_barcode_list_file(walk_dir, dataset_name)
    slide_dict = OrderedDict()
    slide_filetypes = ['jpg', 'mrxs', 'svs', 'tiff', 'isyntax', 'ndpi']
    slide_ind = -1
    for root, subdirs, files in os.walk(walk_dir):
        if 'SegData' in root:
            continue
        print('--\nfolder = ' + root)
        slide_files = [file for file in files if file.split('.')[-1] in slide_filetypes]

        for file in slide_files:
            slide_ind += 1
            slide_dict[slide_ind] = {'dir': root, 'file': file}

    slide_list_df = pd.DataFrame.from_dict(slide_dict, "index")
    slide_list_df.to_excel(excel_file)


def add_barcodes_to_slide_list(data_dir, dataset_name, save_label_image=True, slides_contain_barcodes=True):
    barcode_convention = get_barcode_convention(data_dir)
    if barcode_convention == BARCODE_CONVENTION_NONE:
        return
    print('adding barcodes to slide list')

    start_time = time.time()
    prev_time = start_time

    excel_file = get_barcode_list_file(data_dir, dataset_name)
    slide_list_df = pd.read_excel(excel_file)
    num_slides = len(slide_list_df)
    barcode_list, comment_list = np.empty(num_slides, dtype='S'), np.empty(num_slides, dtype='S')
    barcode_list[:], comment_list[:] = "", ""

    for slide_info in tqdm(slide_list_df.iterrows()):
        slide_ind = slide_info[0]
        print('extracting barcode from slide: ' + slide_info[1]['file'])
        barcode = EMPTY_BARCODE_LIST
        if slides_contain_barcodes and save_label_image:
            barcode = extract_barcode_from_slide(slide_info, barcode_convention)

        barcode_list[slide_ind], comment_list[slide_ind] = add_barcode_to_barcode_list(barcode, save_label_image, data_dir, slide_info)

        if slide_ind % 100 == 0:  # save every 100 slides
            save_barcode_list(slide_list_df, barcode_list, comment_list, excel_file)

        image_time = time.time()
        print('processing time: ' + str(image_time - prev_time) + ' sec')
        prev_time = image_time

    save_barcode_list(slide_list_df, barcode_list, comment_list, excel_file)
    print('Finished, total time: ' + str(time.time() - start_time) + ' sec')


def add_barcode_to_barcode_list(barcode, save_label_image, data_dir, slide_info):
    comment = ''
    n_barcodes = len(barcode)
    if n_barcodes > 1:
        barcode = ",".join(barcode)
    elif n_barcodes == 0:
        if save_label_image:
            comment = save_slide_label_to_image(data_dir, slide_info)
        else:
            comment = 'unreadable label, no thumbnail saved'

    return barcode, comment


def save_barcode_list(slide_data_df, barcode_list, comment_list, excel_file):
    slide_data_df['SlideID'] = barcode_list
    slide_data_df['unreadable'] = comment_list
    slide_data_df.to_excel(excel_file)  # save after every image


def save_slide_label_to_image(data_dir, slide_info):
    print('no barcodes found! saving label to image')
    try:
        label_im = get_slide_label_image(slide_info)
        plt.imshow(label_im)
        label_name = str(slide_info[0]).zfill(4) + slide_info[1]['dir'].replace('/', '_') + '_' + os.path.splitext(slide_info[1]['file'])[0] + '.png'

        if not os.path.isdir(os.path.join(data_dir, 'unreadable_labels')):
            os.mkdir(os.path.join(data_dir, 'unreadable_labels'))
        plt.savefig(os.path.join(data_dir, 'unreadable_labels', label_name))
        comment = 'unreadable label'
    except:
        comment = 'cannot open slide'
    return comment

def extract_barcode_from_slide(slide_info, barcode_convention):
    downsmple_factor = 0.1
    slide_barcode = EMPTY_BARCODE_LIST

    try:
        while slide_barcode == EMPTY_BARCODE_LIST and downsmple_factor <= 1:
            slide_barcode = decode_slide_barcode(slide_info, downsmple_factor=downsmple_factor, barcode_convention=barcode_convention)
            downsmple_factor *= 2
            if 0.5 < downsmple_factor < 1:
                downsmple_factor = 1
    except:
        print('failed during get_slide_label')

    return slide_barcode


def decode_slide_barcode(slide_info, downsmple_factor, barcode_convention):
    slide_barcode = EMPTY_BARCODE_LIST
    label_im = get_slide_label_image(slide_info)
    label_im = resample_slide_label_image(label_im, downsmple_factor)

    all_barcodes = decode(label_im)

    for code in all_barcodes:
        barcode = adjust_barcode_using_convention(barcode=code.data.decode('UTF-8'), barcode_convention=barcode_convention)
        if check_barcode_validity(barcode):
            slide_barcode.append(barcode)

    return slide_barcode


def get_slide_label_image(slide_info):
    slide = openslide.OpenSlide(os.path.join(slide_info[1]['dir'], slide_info[1]['file']))
    label_im = slide.associated_images['label']
    return label_im


def resample_slide_label_image(label_im, downsmple_factor):
    if downsmple_factor != 1:
        w, h = label_im.size
        label_im = label_im.resize((int(w * downsmple_factor), int(h * downsmple_factor)), resample=Image.BICUBIC)
    return label_im


def get_barcode_list_file(main_data_dir, dataset_name):
    return os.path.join(main_data_dir, 'barcode_list_' + dataset_name + '.xlsx')


def get_barcode_convention(data_dir):
    if 'Carmel' in data_dir:
        barcode_convention = BARCODE_CONVENTION_CARMEL
    elif 'Haemek' in data_dir:
        barcode_convention = BARCODE_CONVENTION_HAEMEK
    else:
        barcode_convention = BARCODE_CONVENTION_NONE
    return barcode_convention


def check_barcode_validity(barcode):
    if re.search("\d{2}-\d+/\d+/\d+/\D{1}\Z", barcode) != None:
        return True
    else:
        return False


def adjust_barcode_using_convention(barcode, barcode_convention):
    if len(barcode) < 10 or len(barcode) > 16:
        return EMPTY_BARCODE_LIST
    if barcode_convention == BARCODE_CONVENTION_CARMEL:
        barcode = str(int(barcode[0:4]) - 9788) + '-' + barcode[4:]
    elif barcode_convention == BARCODE_CONVENTION_HAEMEK:
        year = int(barcode[4]) + 14
        barcode = str(year) + '-' + barcode[5:]
    return barcode