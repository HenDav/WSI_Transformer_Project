import numpy as np
from PIL import Image
import os
import glob
import ast

def get_her2_control_dir(data_dir):
    control_dir = os.path.join(data_dir, 'thumbs_w_control_tissue_marked')
    if os.path.isdir(control_dir):
        return control_dir
    else:
        return ''


def remove_slide_artifacts_rows_from_segmentation(img_arr, remove_upper_part_ratio=0.1):
    # remove upper 10% of slide
    h = img_arr.shape[0]
    N_rows_to_remove = int(h*remove_upper_part_ratio)
    img_arr[:N_rows_to_remove, :] = 255
    return img_arr


def remove_control_tissue_rows_according_to_marked_file(img_arr, matching_marked_img_file, color):
    if color is None:
        color = (255, 0, 0)
    else:
        color = ast.literal_eval(color)
    # red_color2 = (237, 27, 36)
    # red_color3 = (227, 24, 44)
    matching_marked_img = Image.open(matching_marked_img_file[0])
    matching_marked_img_arr_resized = np.array(matching_marked_img.resize(img_arr.shape[-2::-1]))

    if matching_marked_img_arr_resized.shape[2] == 4:
        matching_marked_img_arr_resized = matching_marked_img_arr_resized[:, :, 0:3]

    red_mask1 = np.all(matching_marked_img_arr_resized == color, axis=2)
    # red_mask2 = np.all(matching_marked_img_arr_resized == red_color2, axis=2)
    # red_mask3 = np.all(matching_marked_img_arr_resized == red_color3, axis=2)
    print('np.sum(red_mask1):', str(np.sum(red_mask1)))  # temp
    #print('np.sum(red_mask2):', str(np.sum(red_mask2)))  # temp
    #print('np.sum(red_mask3):', str(np.sum(red_mask3)))  # temp
    # red_mask = np.logical_or(np.logical_or(red_mask1, red_mask2), red_mask3)
    red_rows = np.any(red_mask1, axis=1)
    print('np.sum(red_rows):', str(np.sum(red_rows))) #temp
    img_arr[red_rows, :] = 255
    return img_arr


def remove_control_tissue_rows_from_segmentation(img_array, slide_name, data_dir, color):
    control_dir = get_her2_control_dir(data_dir)
    if control_dir != '':
        matching_marked_image_file = glob.glob(os.path.join(control_dir, '*' + slide_name + '.jpg'))
        matching_marked_image_file += glob.glob(os.path.join(control_dir, '*' + slide_name + ' *.jpg'))
        matching_marked_image_file += glob.glob(os.path.join(control_dir, '*' + slide_name + '-edit-*.jpg'))
        matching_marked_file_found = len(matching_marked_image_file) == 1
        if matching_marked_file_found:
            print('Removing control tissue')
            img_array = remove_control_tissue_rows_according_to_marked_file(img_array, matching_marked_image_file, color)
        else:
            print('no matching marked image for slide ' + slide_name)
    return img_array


def avoid_control_tissue_bottom(thumb, bottom_percent=0.4):
    # detect edges of tissue, dump lower part
    thumb_arr = np.array(thumb)
    thumb_binary_inverse = 255 - np.max(thumb_arr, axis=2)
    positions = np.nonzero(thumb_binary_inverse)
    top = positions[0].min()
    bottom = positions[0].max()
    cutoff = int(top + (bottom - top) * (1 - bottom_percent))
    thumb_arr[cutoff:, :, :] = 255
    thumb = Image.fromarray(thumb_arr)
    return thumb


def remove_control_tissue_according_to_dataset(img, is_IHC_slide, slide_name, dataset, data_dir, color):
    # Avoid control tissue on segmentation
    is_porto_pdl1_w_control_tissue = (is_IHC_slide and dataset == 'PORTO_PDL1' and slide_name[-5:] == ' pdl1')
    print(f'dataset[:4].casefold(): {dataset[:4]}')
    is_her2 = dataset[:4].casefold() == 'HER2'.casefold()
    if is_porto_pdl1_w_control_tissue:
        # PORTO second batch IHC contains control tissue
        # identified with " pdl1" in the filename
        img = avoid_control_tissue_bottom(img)
    elif is_her2:
        img = Image.fromarray(remove_control_tissue_rows_from_segmentation(img_array=np.array(img),
                                                                           slide_name=slide_name,
                                                                           data_dir=data_dir,
                                                                           color=color))
    return img


def remove_slide_artifacts_according_to_dataset(img, dataset):
    if dataset[:4].casefold() == 'HER2'.casefold():
        img = Image.fromarray(remove_slide_artifacts_rows_from_segmentation(np.array(img)))
    return img