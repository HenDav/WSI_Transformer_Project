import os
from Dataset_Maker import slide_walker


def add_slide_rename_to_barcode_list(in_dir, Dataset):
    excel_file = slide_walker.get_barcode_list_file(in_dir, Dataset)
    barcode_list = slide_walker.open_barcode_list_file(excel_file)
    barcode_list['slide rename'] = barcode_list['SlideID'].replace('/','_')
    barcode_list.sort_values(by='slide rename')
    prev_id = ""
    for row in barcode_list.iterrows():
        if row['slide rename'] == prev_id:
            row['slide rename'] = row['slide rename'] + "-0"
        prev_id = row['slide rename']

    print('aa')

def rename_slides_according_to_list(in_dir, Dataset):
    out_dir = define_output_dir(in_dir, Dataset)
    excel_file = slide_walker.get_barcode_list_file(in_dir, Dataset)
    barcode_list = slide_walker.open_barcode_list_file(excel_file)

    for row in barcode_list.iterrows():
        try:
            rename_slide_file_and_folder(slide_data=row[1], out_dir=out_dir)
        except KeyError as e:
            print(e)


def rename_slide_file_and_folder(slide_data, out_dir):
    fn = os.path.join(slide_data['dir'], slide_data['file'])
    print('file:', fn)
    if slide_data['slide rename'] == -1:
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
