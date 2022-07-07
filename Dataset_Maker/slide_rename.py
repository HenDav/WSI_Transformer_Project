import os

from Dataset_Maker import dataset_utils
from Dataset_Maker import slide_walker


UKNOWN_SLIDE_ID = '-1'


def add_slide_rename_to_barcode_list(in_dir, Dataset):
    excel_file = slide_walker.get_barcode_list_file(in_dir, Dataset)
    barcode_list = dataset_utils.open_excel_file(excel_file)
    barcode_list.at[barcode_list['SlideID'].isna(), 'SlideID'] = UKNOWN_SLIDE_ID
    barcode_list['slide rename'] = [barcode.replace('/', '_') for barcode in barcode_list['SlideID']]
    barcode_list = barcode_list.sort_values(by='slide rename')
    prev_id = ""
    add_suffix = 0
    for i_row, row in barcode_list.iterrows():
        if row['slide rename'] == prev_id:
            barcode_list.at[i_row, 'slide rename'] += "-" + str(add_suffix)
            add_suffix += 1
        else:
            add_suffix = 0
            prev_id = row['slide rename']
    dataset_utils.save_df_to_excel(barcode_list, excel_file)
    print('added slide_rename column to barcode list')


def rename_slides_according_to_list(in_dir, Dataset):
    out_dir = define_output_dir(in_dir, Dataset)
    excel_file = slide_walker.get_barcode_list_file(in_dir, Dataset)
    barcode_list = dataset_utils.open_excel_file(excel_file)

    for row in barcode_list.iterrows():
        try:
            rename_slide_file_and_folder(slide_data=row[1], out_dir=out_dir)
        except KeyError as e:
            print(e)


def delete_empty_folders(data_dir, Dataset):
    print('removing empty folders for data directory = ' + data_dir)
    walk = list(os.walk(data_dir))
    for path, _, _ in walk[::-1]:
        if (os.path.join(data_dir, Dataset) in path) or ('unreadable_labels' in path):
            continue
        if 'thumbs.db' in os.listdir(path):
            move_thumbs_db_file(data_dir, path)
        if len(os.listdir(path)) == 0:
            try:
                os.remove(path)
            except Exception as E:
                print('could not erase folder ' + path)
                print(E)


def move_thumbs_db_file(data_dir, path):
    if not os.path.isdir(os.path.join(data_dir, 'thumbs_db')):
        os.mkdir(os.path.join(data_dir, 'thumbs_db'))
    new_filename = 'thumbs_' + os.path.basename(path) + '.db'
    orig_name = os.path.join(path, 'thumbs.db')
    os.rename(orig_name, os.path.join(data_dir, 'thumbs_db', new_filename))


def rename_slide_file_and_folder(slide_data, out_dir):
    fn = os.path.join(slide_data['dir'], slide_data['file'])
    print('file:', fn)
    if str(slide_data['slide rename']) == '-1':
        return
    if is_mrxs(fn):
        dn = fn[:-5]
        if os.path.isfile(fn) and os.path.isdir(dn):
            new_dirname = str(slide_data['slide rename'])
            new_filename = new_dirname + '.mrxs'
            rename_file(fn, new_filename, out_dir)
            rename_file(dn, new_dirname, out_dir)
    else:  # svs and other files
        if os.path.isfile(fn):
            new_filename = str(slide_data['slide rename'])
            rename_file(fn, new_filename, out_dir)
    print('renamed successfully')


def rename_file(orig_name, new_name, out_dir):
    os.rename(orig_name, os.path.join(out_dir, new_name))


def is_mrxs(filename):
    return filename.split('.')[-1] == 'mrxs'


def define_output_dir(in_dir, Dataset):
    out_dir = os.path.join(in_dir, Dataset)
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    return out_dir
