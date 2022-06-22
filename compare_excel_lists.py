import pandas as pd
import numpy as np
import os


def find_which_arr1_items_are_missing_in_array2(arr1, arr2):
    missing_in_arr2 = []
    for item1 in arr1:
        if item1 not in arr2:
            missing_in_arr2.append(item1)
    return missing_in_arr2


def save_missing_list_to_excel(dir_name, fn, missing_in_f):
    df_missing_in_f = pd.DataFrame(missing_in_f)
    df_missing_in_f.to_excel(os.path.join(dir_name, 'missing_in_' + fn))
    print('missing in ' + fn + ':')
    print(missing_in_f)


def get_list_with_no_file_extensions(dir_name, fn, header_name):
    df = pd.read_excel(os.path.join(dir_name, fn))
    return np.array([os.path.splitext(str(file))[0] for file in df[header_name]])


def compare_excel_lists(dir_name, fn1, fn2, header_name):
    file_arr = []
    for fn in [fn1, fn2]:
        file_arr.append(get_list_with_no_file_extensions(dir_name, fn, header_name))

    missing_in_f2 = find_which_arr1_items_are_missing_in_array2(file_arr[0], file_arr[1])
    missing_in_f1 = find_which_arr1_items_are_missing_in_array2(file_arr[1], file_arr[0])

    for fn, missing_in_f in zip([fn1, fn2], [missing_in_f1, missing_in_f2]):
        save_missing_list_to_excel(dir_name, fn, missing_in_f)


def merge_excel_lists(dirname, file1, file2, merge_key):
    df = []
    for i, ifile in enumerate([file1, file2]):
        df.append(pd.read_excel(os.path.join(dirname, ifile)))
        df[-1][merge_key] = df[-1][merge_key].astype(str)

    df_m = pd.merge(df[0], df[1], how='outer', on=merge_key)
    df_m.to_excel(os.path.join(dirname, 'merged_file.xlsx'))

def fix_TMA_slide_names_in_label_excel_list(dirname, fn, filename_key):
    df = pd.read_excel(os.path.join(dirname, fn))
    file_list = df[filename_key]
    new_names = []
    for image_fn in file_list:
        image_name = os.path.splitext(image_fn)[0]
        last_number = image_name.split('_')[-1]
        last_number_with_zeros = str(last_number).zfill(3)
        new_image_name = '_'.join(image_name.split('_')[:-1]) + '_' + last_number_with_zeros + os.path.splitext(image_fn)[-1]
        new_names.append(new_image_name)
    df['new_file'] = new_names
    df.to_excel(os.path.join(dirname, fn + '_filename_zfilled.xlsx'))