import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
import os
import glob
from pyzbar.pyzbar import decode as decode_1d  # for 1d barcodes, if needed
import time
import xlsxwriter
from sklearn import linear_model


def add_unreadable_label_images_to_slide_list(data_dir, image_contains_individual_labels=True):
    start_time = time.time()
    prev_time = start_time

    img_dir = os.path.join(data_dir, 'unreadable_labels')
    size_factor = 1

    if not os.path.isdir(os.path.join(img_dir, 'out')):
        os.mkdir(os.path.join(img_dir, 'out'))

    img_files_jpeg = glob.glob(os.path.join(img_dir, '*.jpeg'))
    img_files_jpg = glob.glob(os.path.join(img_dir, '*.jpg'))
    img_files_png = glob.glob(os.path.join(img_dir, '*.png'))
    img_files = img_files_jpeg + img_files_jpg + img_files_png
    total_valids = 0
    filenames = []

    for img_file in img_files:
        print('processing image: ' + img_file)
        image = cv2.imread(img_file)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_pil = Image.fromarray(gray)
        w, h = gray.shape
        if size_factor != 1:
            gray_pil2 = gray_pil.resize((h * size_factor, w * size_factor), resample=Image.BICUBIC)
        else:
            gray_pil2 = gray_pil

        img = np.array(gray_pil2)

        if not image_contains_individual_labels:
            fig = plt.figure(frameon=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            # ax.imshow(image, aspect='auto')
        num_valids = 0

        if image_contains_individual_labels:
            # text_image = img[code.rect.top - 150:code.rect.top, code.rect.left - 70:code.rect.left + 220]
            text_image = img[w // 8:9 * w // 20, h // 8:6 * h // 8]
            save_temp_label_image()
            num_valids += 1
            total_valids += 1
            filenames.append(os.path.basename(img_file))
        else:
            codes_1d = decode_1d(img)
            ax.imshow(image, aspect='auto')

            new_try = False
            if new_try:
                # align images, RanS 5.4.22
                ptsA = np.array([code.rect[:2] for code in codes_1d])
                ptsA_sorted = np.zeros_like(ptsA)
                N_codes = len(codes_1d)
                ptsA_sorted_by_y = ptsA[ptsA[:, 1].argsort()]
                row_len = 12
                for ii in np.arange(0, N_codes, row_len):
                    ptsA_row = ptsA_sorted_by_y[ii: ii + row_len]
                    ptsA_row_sorted_by_x = ptsA_row[ptsA_row[:, 0].argsort()]
                    ptsA_sorted[ii: ii + row_len] = ptsA_row_sorted_by_x

                new_grid_x, new_grid_y = np.meshgrid(np.linspace(300, 2000, row_len), np.linspace(300, 1500, 4))
                pts_B = np.c_[new_grid_x.ravel(), new_grid_y.ravel()]
                img = align_image_using_matched_points(ptsA_sorted, pts_B)
                codes_1d = decode_1d(img)

                # estimate grid
                code_x, code_y = get_corrected_grid()
            else:
                # barcode_max_width = np.max([code.rect.width for code in codes_1d])
                # barcode_max_height = np.max([code.rect.height for code in codes_1d])
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
                    # text_image = img[yy - 200:yy + 100, xx - 120:xx + 180]
                    text_image = img[y0: y0 + label_height, x0:x0 + label_width]
                    save_temp_label_image()
                    rect = patches.Rectangle((x0, y0), label_width, label_height, linewidth=2, edgecolor='r',
                                             facecolor='none', alpha=0.6)
                    ax.add_patch(rect)
                    filenames.append(os.path.basename(img_file))
                    num_valids += 1
                    total_valids += 1

        if not image_contains_individual_labels:
            print('found ' + str(len(codes_1d)) + ' barcodes, of which ', str(num_valids), ' are valid')
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
            fig.savefig(os.path.join(img_dir, 'out', os.path.splitext(os.path.basename(img_file))[0] + '_results.jpg'),
                        dpi=300)
            plt.close(fig)
        # df.to_excel(os.path.join(img_dir, 'out', 'barcode_list.xlsx'))  #save after every image
        image_time = time.time()
        print('processing time: ' + str(image_time - prev_time) + ' sec')
        prev_time = image_time

    # write images to workbook
    fn = os.path.basename(img_dir)
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
    if image_contains_individual_labels:
        worksheet.set_default_row(35)
    else:
        worksheet.set_default_row(55)

    for ii in range(total_valids):
        img_file = os.path.join(img_dir, 'out', 'temp_fig' + str(ii) + '.png')
        worksheet.write_string('A' + str(ii + 2), filenames[ii])
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


def save_temp_label_image():
    fig1 = plt.figure()
    plt.imshow(text_image)
    ax1 = plt.gca()
    ax1.axes.xaxis.set_visible(False)
    ax1.axes.yaxis.set_visible(False)
    temp_im_path = os.path.join(img_dir, 'out', 'temp_fig' + str(total_valids) + '.png')
    fig1.savefig(temp_im_path, bbox_inches='tight')
    plt.close()


def estimate_1d_grid_params_with_ransac(N_rows, upper_left_barcode_pos, ax):
    arr = np.array([code.rect[ax] for code in codes_1d])
    good_barcode_loc_mask = np.array([code.rect.width > 10 for code in codes_1d])

    d_estimate = (np.max(arr) - np.min(arr)) / (N_rows - 1)
    arr_for_ransac = arr[good_barcode_loc_mask]
    # Nx = np.round((arr_for_ransac - np.min(arr_for_ransac)) / d_estimate)
    # Nx = np.round((arr_for_ransac - np.min(arr)) / d_estimate)
    Nx = np.round((arr_for_ransac - upper_left_barcode_pos[ax]) / d_estimate)
    # Nx = np.round((arr - arr[0]) / d_estimate)
    # ransac = linear_model.RANSACRegressor()
    ransac = linear_model.RANSACRegressor(residual_threshold=np.median(np.abs(arr - np.median(arr))) / 10)

    ransac.fit(Nx.reshape(-1, 1), arr_for_ransac)
    intercept = ransac.estimator_.intercept_
    d = ransac.estimator_.coef_
    return intercept, d


def get_upper_left_barcode_pos():
    dist_from_origin = np.array([code.rect[0] ** 2 + code.rect[1] ** 2 for code in codes_1d])
    upper_left_barcode_pos = codes_1d[np.argmin(dist_from_origin)].rect
    return upper_left_barcode_pos[:2]


def get_num_non_overlapping_codes():
    num_codes = 1
    x_arr = np.array([code.rect[0] for code in codes_1d])
    y_arr = np.array([code.rect[1] for code in codes_1d])
    for ii in range(1, len(x_arr)):
        min_dist_per_code = np.min(np.linalg.norm((x_arr[ii] - x_arr[:ii], y_arr[ii] - y_arr[:ii]), axis=0))
        new_barcode = (min_dist_per_code > barcode_width)
        if new_barcode:
            num_codes += 1
    return num_codes


def estimate_grid_params():
    # N_codes = len(codes_1d)
    N_codes = get_num_non_overlapping_codes()
    N_columns = 12
    N_rows = int(np.ceil(N_codes / N_columns))
    upper_left_barcode_pos = get_upper_left_barcode_pos()
    # if N_rows>0:
    y0, dy = estimate_1d_grid_params_with_ransac(N_rows, upper_left_barcode_pos, ax=1)
    x0, dx = estimate_1d_grid_params_with_ransac(N_columns, upper_left_barcode_pos, ax=0)
    return (x0, y0), (dx, dy), N_rows


def estimate_grid_points(x0, dx, y0, dy):
    x_arr, y_arr = [], []
    for code in codes_1d:
        y_arr.append(code.rect[1])
        barcode_left_corner = code.rect.width > 10
        if barcode_left_corner:
            x_arr.append(code.rect[0])
        else:
            x_arr.append(code.rect[0] - barcode_width)
    min_Nx = np.round((np.min(x_arr) - x0) / dx)
    max_Nx = np.round((np.max(x_arr) - x0) / dx)
    if dy > 0:
        min_Ny = np.round((np.min(y_arr) - y0) / dy)
        max_Ny = np.round((np.max(y_arr) - y0) / dy)
    else:
        min_Ny, max_Ny = 0, 0
    # Nx, Ny = np.meshgrid(np.arange(0, 12, 1), np.arange(0, N_rows, 1))
    Nx, Ny = np.meshgrid(np.arange(min_Nx, max_Nx + 1, 1), np.arange(min_Ny, max_Ny + 1, 1))
    Nx, Ny = Nx.flatten(), Ny.flatten()
    return Nx, Ny


def get_corrected_grid():
    (x0, y0), (dx, dy), N_rows = estimate_grid_params()
    Nx, Ny = estimate_grid_points(x0, dx, y0, dy)

    new_y = Ny * dy + y0
    new_x = Nx * dx + x0

    return (new_x, new_y)


# based on https://pyimagesearch.com/2020/08/31/image-alignment-and-registration-with-opencv/
def align_image_using_matched_points(ptsA, ptsB):
    # compute the homography matrix between the two sets of matched points
    (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
    # use the homography matrix to align the images
    (h, w) = image.shape[:2]
    aligned = cv2.warpPerspective(image, H, (w, h))
    # return the aligned image
    return aligned


def check_barcode_position(top, left, code):
    #w_margin = int(barcode_med_width/2)
    w_margin = int(barcode_med_width / 1.5)
    h_margin = int(barcode_med_height)
    top_start = max(0, top - h_margin)
    left_start = max(0, left - w_margin)
    barcode_img = img[top_start:top + barcode_med_height + h_margin,
                       left_start:left + barcode_med_width + w_margin]
    barcode_check = decode_1d(barcode_img)
    #plt.figure()
    #plt.imshow(barcode_img)
    if len(barcode_check) and barcode_check[0].data == code:
        return 1
    else:
        return 0


def correct_barcode_position():
    code_x, code_y = [], []
    #ii = 0
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


