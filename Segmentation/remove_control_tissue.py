import numpy as np
from PIL import Image
import os, sys
import glob


def get_her2_control_dir(dataset):
    if dataset == 'HER2_1':
        if sys.platform == 'linux':  # GIPdeep
            control_dir = r'/mnt/gipmed_new/Data/Breast/Carmel/Her2/Batch_1/thumbs_w_control_marked_in_red'
        elif sys.platform == 'win32':  # GIPdeep
            control_dir = r'C:\Users\User\OneDrive - Technion\Her2_batch1_w_control_review\marked'
    else:
        raise ValueError('marked image folder not defined for dataset ' + dataset)
    return control_dir


def remove_slide_artifacts_rows_from_segmentation(img_arr, remove_upper_part_ratio=0.1):
    #remove upper 10% of slide
    h = img_arr.shape[0]
    N_rows_to_remove = int(h*remove_upper_part_ratio)
    img_arr[:N_rows_to_remove, :] = 255
    return img_arr


def remove_control_tissue_rows_according_to_marked_file(img_arr, matching_marked_img_file):
    red_color1 = (231, 27, 28)
    red_color2 = (237, 27, 36)
    matching_marked_img = Image.open(matching_marked_img_file[0])
    matching_marked_img_arr_resized = np.array(matching_marked_img.resize(img_arr.shape[-2::-1]))
    red_mask1 = np.all(matching_marked_img_arr_resized == red_color1, axis=2)
    red_mask2 = np.all(matching_marked_img_arr_resized == red_color2, axis=2)
    red_mask = np.logical_or(red_mask1, red_mask2)
    red_rows = np.any(red_mask, axis=1)
    img_arr[red_rows, :] = 255
    return img_arr


def remove_control_tissue_rows_from_segmentation(img_array, slide_name, dataset):
    control_dir = get_her2_control_dir(dataset)
    matching_marked_image_file = glob.glob(os.path.join(control_dir, '*' + slide_name + '.jpg'))
    matching_marked_file_found = len(matching_marked_image_file) == 1
    if matching_marked_file_found:
        img_array = remove_control_tissue_rows_according_to_marked_file(img_array, matching_marked_image_file)
    else:
        print('no matching marked image for slide ' + slide_name)
    return img_array


'''
def remove_control_tissue_from_segmentation(img_array, red_color=(231, 27, 28)):
    #this is a preparation for a more complete implementation, removing only what's inside the red circle
    red_mask = np.all(img_array == red_color, axis=2)
    red_mask *= 255
    red_mask = red_mask.astype(np.uint8)
    kernel_size = 100
    kernel_smooth = np.ones((kernel_size, kernel_size), dtype=np.float32) / kernel_size ** 2
    red_mask_filt = cv2.filter2D(red_mask, -1, kernel_smooth)
    red_mask_filt[red_mask_filt<255] = 0
    contours, _ = cv2.findContours(red_mask_filt, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #to be completed
'''


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


def remove_control_tissue_according_to_dataset(img, is_IHC_slide, slide_name, dataset):
    # Avoid control tissue on segmentation
    if (is_IHC_slide and dataset == 'PORTO_PDL1' and slide_name[-5:] == ' pdl1'):
        # PORTO second batch IHC contains control tissue
        #identified with " pdl1" in the filename
        img = avoid_control_tissue_bottom(img)
    elif dataset[:4] == 'HER2':
        img = Image.fromarray(remove_control_tissue_rows_from_segmentation(img_array=np.array(img), slide_name=slide_name, dataset=dataset))
    return img


def remove_slide_artifacts_according_to_dataset(img, dataset):
    if dataset[:4] == 'HER2':
        img = Image.fromarray(remove_slide_artifacts_rows_from_segmentation(np.array(img)))
    return img