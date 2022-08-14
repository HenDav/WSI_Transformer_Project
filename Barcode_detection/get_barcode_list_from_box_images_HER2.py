import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
import os
import glob
import pandas as pd
from pyzbar.pyzbar import decode as decode_1d
import time
import argparse
import xlsxwriter
from Barcode_detection.get_barcode_list_from_box_images import get_box_name
from Dataset_Maker.dataset_utils import format_empty_spaces_as_string

parser = argparse.ArgumentParser(description='Barcode_detection')
parser.add_argument('-in', '--img_dir', type=str, default=r'C:\ran_data\RAMBAM\AH7-AH8_taken', help='input directory')
parser.add_argument('--image_contains_individual_labels', action='store_true', help='images of individual labels')
args = parser.parse_args()


def save_temp_label_image():
    fig1 = plt.figure()
    plt.imshow(text_image)
    ax1 = plt.gca()
    ax1.axes.xaxis.set_visible(False)
    ax1.axes.yaxis.set_visible(False)
    temp_im_path = os.path.join(img_dir, 'out', 'temp_fig' + str(total_valids) + '.png')
    fig1.savefig(temp_im_path, bbox_inches='tight')
    plt.close()


def check_barcode_position(top, left, code):
    w_margin = int(barcode_med_width / 1.5)
    h_margin = int(barcode_med_height)
    top_start = max(0, top - h_margin)
    left_start = max(0, left - w_margin)
    barcode_img = img[top_start:top + barcode_med_height + h_margin,
                      left_start:left + barcode_med_width + w_margin]
    barcode_check = decode_1d(barcode_img)
    if len(barcode_check) and barcode_check[0].data == code:
        return 1
    else:
        return 0


def correct_barcode_position():
    code_x, code_y = [], []
    for code in codes_1d:
        barcode_caught_on_left = code.rect.width > 10
        barcode_caught_on_top = code.rect.height > 10
        if barcode_caught_on_left and barcode_caught_on_top:
            code_x.append(code.rect.left)
            code_y.append(code.rect.top)
        else:
            orig_position_barcode = check_barcode_position(code.rect.top, code.rect.left, code.data)
            if orig_position_barcode:
                code_x.append(code.rect.left)
                code_y.append(code.rect.top)
            else:
                left_position_barcode = check_barcode_position(code.rect.top, code.rect.left - barcode_med_width, code.data)
                if left_position_barcode:
                    code_x.append(code.rect.left - barcode_med_width)
                    code_y.append(code.rect.top)
                else:
                    print('failed to validate barcode position')

    return code_x, code_y


if __name__ == '__main__':
    start_time = time.time()
    prev_time = start_time

    img_dir = args.img_dir
    size_factor = 1

    if not os.path.isdir(os.path.join(img_dir, 'out')):
        os.mkdir(os.path.join(img_dir, 'out'))

    img_files_jpeg = glob.glob(os.path.join(img_dir, '*.jpeg'))
    img_files_jpg = glob.glob(os.path.join(img_dir, '*.jpg'))
    img_files_png = glob.glob(os.path.join(img_dir, '*.png'))
    img_files = img_files_jpeg + img_files_jpg + img_files_png
    df = pd.DataFrame(columns=['file', 'index', 'Box', 'SlideID', 'Comments'])
    total_valids = 0
    filenames, boxnames = [], []

    for img_file in img_files:
        print('processing image: ' + img_file)
        image = cv2.imread(img_file)
        box = get_box_name(img_file)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_pil = Image.fromarray(gray)
        w, h = gray.shape
        if size_factor != 1:
            gray_pil2 = gray_pil.resize((h * size_factor, w * size_factor), resample=Image.BICUBIC)
        else:
            gray_pil2 = gray_pil

        img = np.array(gray_pil2)
        h_img = image.shape[0]

        if not args.image_contains_individual_labels:
            fig = plt.figure(frameon=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
        num_valids = 0

        if args.image_contains_individual_labels:
            text_image = img[w // 8:9 * w // 20, h // 8:6 * h // 8]
            save_temp_label_image()
            num_valids += 1
            total_valids += 1
            filenames.append(os.path.basename(img_file))
        else:
            codes_1d = decode_1d(img)
            ax.imshow(image, aspect='auto')

            barcode_med_width = int(np.median([code.rect.width for code in codes_1d]))
            barcode_med_height = int(np.median([code.rect.height for code in codes_1d]))
            code_x, code_y = correct_barcode_position()
            label_x_shift = int(barcode_med_width / 2)
            label_y_shift = int(barcode_med_height * 2.5)
            label_height = int(barcode_med_height * 10)
            label_width = int(barcode_med_width * 2.25)

            for x_code, y_code in zip(code_x, code_y):
                xx, yy = int(x_code), int(y_code)
                label_is_outside_image = (xx - label_x_shift) < -(0.2 * label_width)
                if not label_is_outside_image:
                    x0 = np.maximum(0, xx - label_x_shift)
                    y0 = np.maximum(0, yy - label_y_shift)
                    text_image = img[y0: y0 + label_height, x0:x0 + label_width]
                    save_temp_label_image()
                    rect = patches.Rectangle((x0, y0), label_width, label_height, linewidth=2, edgecolor='r',
                                             facecolor='none', alpha=0.6)
                    ax.add_patch(rect)
                    filenames.append(os.path.basename(img_file))
                    boxnames.append(box)
                    num_valids += 1
                    total_valids += 1

        if not args.image_contains_individual_labels:
            print('found ' + str(len(codes_1d)) + ' barcodes, of which ', str(num_valids), ' are valid')
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
            fig.savefig(os.path.join(img_dir, 'out', os.path.splitext(os.path.basename(img_file))[0] + '_results.jpg'),
                        dpi=300)
            plt.close(fig)
        image_time = time.time()
        print('processing time: ' + str(image_time - prev_time) + ' sec')
        prev_time = image_time

    # write images to workbook
    fn = os.path.basename(args.img_dir)
    workbook = xlsxwriter.Workbook(os.path.join(img_dir, 'out', 'barcode_list_images_' + fn + '.xlsx'))
    worksheet = workbook.add_worksheet()
    worksheet.write_string('A1', 'file')
    worksheet.write_string('B1', 'Box')
    worksheet.write_string('H1', 'barcode image')
    worksheet.write_string('C1', 'Year')
    worksheet.write_string('D1', 'SampleID')
    worksheet.write_string('E1', 'TissueID')
    worksheet.write_string('F1', 'BlockID')
    worksheet.write_string('G1', 'SlideID')
    worksheet.write_string('I1', 'Comments')
    worksheet.write_string('J1', 'barcode (auto)')
    if args.image_contains_individual_labels:
        worksheet.set_default_row(35)
    else:
        worksheet.set_default_row(55)

    for ii in range(total_valids):
        img_file = os.path.join(img_dir, 'out', 'temp_fig' + str(ii) + '.png')
        worksheet.write_string('A' + str(ii + 2), filenames[ii])
        worksheet.write_string('B' + str(ii + 2), boxnames[ii])
        worksheet = format_empty_spaces_as_string(workbook, worksheet, ii, ['C', 'D', 'E', 'F', 'G'])
        worksheet.insert_image('H' + str(ii + 2), img_file, {'x_scale': 0.2, 'y_scale': 0.2})
        formula_string = '=CONCATENATE(C' + str(ii + 2) + ',"-",D' + str(ii + 2) + ',"/",E' + str(
            ii + 2) + ',"/",F' + str(ii + 2) + ',"/",G' + str(ii + 2) + ')'
        worksheet.write_formula('J' + str(ii + 2), formula_string)

    worksheet.set_zoom(200)
    workbook.close()

    for ii in range(total_valids):
        img_file = os.path.join(img_dir, 'out', 'temp_fig' + str(ii) + '.png')
        os.remove(img_file)

    print('Finished, total time: ' + str(time.time() - start_time) + ' sec')
