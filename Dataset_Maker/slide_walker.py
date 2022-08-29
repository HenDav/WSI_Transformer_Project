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
import xlsxwriter
from Dataset_Maker import dataset_utils
import numpy as np


BARCODE_CONVENTION_NONE = 0
BARCODE_CONVENTION_CARMEL = 1
BARCODE_CONVENTION_HAEMEK = 2


def create_slide_list(walk_dir, dataset_name):
    print('creating slide list for data directory = ' + walk_dir)

    barcode_list_file = get_barcode_list_file(walk_dir, dataset_name)
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
    dataset_utils.save_df_to_excel(slide_list_df, barcode_list_file)


def add_barcodes_to_slide_list(data_dir, dataset_name, scan_barcodes=True):
    barcode_convention = get_barcode_convention(data_dir)
    if barcode_convention == BARCODE_CONVENTION_NONE:
        return
    print('adding barcodes to slide list')

    start_time = time.time()
    prev_time = start_time

    barcode_list_file = get_barcode_list_file(data_dir, dataset_name)
    slide_list_df = dataset_utils.open_excel_file(barcode_list_file)
    barcode_list, comment_list, label_image_name_list = [], [], []

    for slide_info in tqdm(slide_list_df.iterrows()):
        # slide_ind = slide_info[0]
        print('extracting barcode from slide: ' + slide_info[1]['file'])
        barcode = []
        if scan_barcodes:
            barcode = extract_barcode_from_slide(slide_info, barcode_convention)

        label_image_name, comment = extract_slide_label(data_dir, slide_info, len(barcode))
        barcode = parse_barcode(barcode)

        barcode_list.append(barcode)
        comment_list.append(comment)
        label_image_name_list.append(label_image_name)

        image_time = time.time()
        print('processing time: ' + str(image_time - prev_time) + ' sec')
        prev_time = image_time

    save_barcode_list(slide_list_df, barcode_list, comment_list, barcode_list_file)
    write_label_images_to_excel(slide_list_df, label_image_name_list, data_dir, dataset_name)
    print('Finished, total time: ' + str(time.time() - start_time) + ' sec')


def merge_manual_barcodes_to_barcode_list(data_dir, dataset_name):
    manual_barcodes_file = get_manual_barcode_file(data_dir, dataset_name, extension='_fixed')
    manual_barcodes_df = dataset_utils.open_excel_file(manual_barcodes_file)
    manual_barcodes_df.loc[manual_barcodes_df['barcode (auto)'] == '-///', 'barcode (auto)'] = np.nan
    N_manual = len(manual_barcodes_df) \
               - manual_barcodes_df['barcode (auto)'].isna().sum() \
               - (manual_barcodes_df['barcode (auto)'] == 0).sum()

    if N_manual:
        barcode_list_file = get_barcode_list_file(data_dir, dataset_name)
        slide_list_df = dataset_utils.open_excel_file(barcode_list_file)
        slide_list_df['SlideID'] = slide_list_df['SlideID'].combine_first(manual_barcodes_df['barcode (auto)'])
        dataset_utils.save_df_to_excel(slide_list_df, barcode_list_file)
        print('merged ' + str(N_manual) + ' manual barcodes into barcode list')


def write_label_images_to_excel(slide_list_df, label_image_name_list, data_dir, dataset_name):
    img_dir = os.path.join(data_dir, 'unreadable_labels')
    # write images to workbook
    manual_barcodes_file = get_manual_barcode_file(data_dir, dataset_name)
    workbook = xlsxwriter.Workbook(manual_barcodes_file)
    worksheet = workbook.add_worksheet()
    worksheet.write_string('A1', 'dir')
    worksheet.write_string('B1', 'file')
    worksheet.write_string('C1', 'box')
    worksheet.write_string('I1', 'barcode image')
    worksheet.write_string('D1', 'Year')
    worksheet.write_string('E1', 'SampleID')
    worksheet.write_string('F1', 'TissueID')
    worksheet.write_string('G1', 'BlockID')
    worksheet.write_string('H1', 'SlideID')
    worksheet.write_string('J1', 'barcode (auto)')
    worksheet.set_default_row(35)

    for ii, img in enumerate(label_image_name_list):
        img_file = os.path.join(img_dir, img)
        worksheet.write_string('A' + str(ii + 2), slide_list_df['dir'][ii])
        worksheet.write_string('B' + str(ii + 2), slide_list_df['file'][ii])
        worksheet = dataset_utils.format_empty_spaces_as_string(workbook, worksheet, ii, ['D', 'E', 'F', 'G', 'H'])
        if img != '':
            worksheet.insert_image('I' + str(ii + 2), img_file, {'x_scale': 0.12, 'y_scale': 0.12})
            formula_string = '=CONCATENATE(D' + str(ii + 2) + ',"-",E' + str(ii + 2) + ',"/",F' + str(
                ii + 2) + ',"/",G' + str(ii + 2) + ',"/",H' + str(ii + 2) + ')'
            worksheet.write_formula('I' + str(ii + 2), formula_string)

    worksheet.set_zoom(200)
    workbook.close()


def parse_barcode(barcode):
    n_barcodes = len(barcode)
    if n_barcodes == 0:
        barcode = ''
    elif n_barcodes == 1:
        barcode = barcode[0]
    else:
        barcode = ','.join(barcode)
    return barcode


def save_barcode_list(slide_list_df, barcode_list, comment_list, excel_file):
    barcode_list = pad_list_with_empty_strings(barcode_list, len(slide_list_df))
    comment_list = pad_list_with_empty_strings(comment_list, len(slide_list_df))
    slide_list_df['SlideID'] = barcode_list
    slide_list_df['unreadable'] = comment_list
    dataset_utils.save_df_to_excel(slide_list_df, excel_file)


def pad_list_with_empty_strings(l, desired_len):
    l += [''] * (desired_len - len(l))
    return l


def extract_slide_label(data_dir, slide_info, n_barcodes):
    if n_barcodes == 0:
        label_image, comment = save_slide_label_to_image(data_dir, slide_info)
    else:
        label_image, comment = '', ''
    return label_image, comment


def save_slide_label_to_image(data_dir, slide_info):
    print('no barcodes found! saving label to image')
    try:
        label_im = get_slide_label_image(slide_info)
        fig1 = plt.figure()
        plt.imshow(label_im)
        dir_str = slide_info[1]['dir'].replace('\\', '_').replace('/', '_').replace(':', '_')
        label_name = str(slide_info[0]).zfill(4) + '_' + dir_str + '_' + os.path.splitext(slide_info[1]['file'])[
            0] + '.png'

        if not os.path.isdir(os.path.join(data_dir, 'unreadable_labels')):
            os.mkdir(os.path.join(data_dir, 'unreadable_labels'))

        ax1 = plt.gca()
        ax1.axes.xaxis.set_visible(False)
        ax1.axes.yaxis.set_visible(False)
        fig1.savefig(os.path.join(data_dir, 'unreadable_labels', label_name), bbox_inches='tight')
        plt.close()
        comment = 'unreadable label'
    except:
        comment = 'cannot open slide'
        label_name = ''
    return label_name, comment


def extract_barcode_from_slide(slide_info, barcode_convention):
    downsmple_factor = 0.1
    slide_barcode = []
    try:
        while slide_barcode == [] and downsmple_factor <= 1:
            slide_barcode = decode_slide_barcode(slide_info, downsmple_factor=downsmple_factor,
                                                 barcode_convention=barcode_convention)
            downsmple_factor *= 2
            if 0.5 < downsmple_factor < 1:
                downsmple_factor = 1
    except Exception as E:
        raise E
    return slide_barcode


def decode_slide_barcode(slide_info, downsmple_factor, barcode_convention):
    slide_barcode = []
    label_im = get_slide_label_image(slide_info)
    label_im = resample_slide_label_image(label_im, downsmple_factor)
    all_barcodes = decode(label_im)
    for code in all_barcodes:
        barcode = adjust_barcode_using_convention(barcode=code.data.decode('UTF-8'),
                                                  barcode_convention=barcode_convention)
        if check_barcode_validity(barcode):
            slide_barcode.append(barcode)

    return slide_barcode


def get_slide_label_image(slide_info):
    slide = openslide.OpenSlide(os.path.join(slide_info[1]['dir'], slide_info[1]['file']))
    if 'label' in slide.associated_images._keys():
        label_im = slide.associated_images['label']
    else:
        raise IOError('Slide has no label image')
    return label_im


def resample_slide_label_image(label_im, downsmple_factor):
    if downsmple_factor != 1:
        w, h = label_im.size
        label_im = label_im.resize((int(w * downsmple_factor), int(h * downsmple_factor)), resample=Image.BICUBIC)
    return label_im


def get_barcode_list_file(main_data_dir, dataset_name):
    return os.path.join(main_data_dir, 'barcode_list_' + dataset_name + '.xlsx')


def get_manual_barcode_file(data_dir, dataset_name, extension=''):
    img_dir = os.path.join(data_dir, 'unreadable_labels')
    manual_barcodes_file = os.path.join(img_dir, 'barcode_list_images_' + dataset_name + extension + '.xlsx')
    return manual_barcodes_file


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
        return []
    if barcode_convention == BARCODE_CONVENTION_CARMEL:
        if int(barcode[:2]) > 90:  # old format, adjustment required
            year = int(barcode[0:4]) - 9788
            if year > 200:  # year 21 shows as 201, and so on
                year -= 180
            barcode = str(year) + '-' + barcode[4:]
    elif barcode_convention == BARCODE_CONVENTION_HAEMEK:
        year = int(barcode[4]) + 14
        barcode = str(year) + '-' + barcode[5:]
    return barcode
