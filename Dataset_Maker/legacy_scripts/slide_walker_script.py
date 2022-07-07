import os
import openslide
import matplotlib.pyplot as plt
from pylibdmtx.pylibdmtx import decode as decode
import pandas as pd
import time
from PIL import Image
import re, argparse
from collections import OrderedDict

parser = argparse.ArgumentParser(description='WSI_REG Slide inference')
parser.add_argument('-wd', '--walk_dir', type=str, default=r'/mnt/gipmed_new/Data/BoneMarrow/new_slides_041021', help='input walk dir')
parser.add_argument('--has_barcodes', action='store_true', help='scan label for barcodes')
parser.add_argument('--save_labels', action='store_true', help='print barcode/label image')
parser.add_argument('--is_haemek', action='store_true', help='data is not from Carmel')
parser.add_argument('--is_jpg', action='store_true', help='data is in jpg format (TMA)')
args = parser.parse_args()
is_carmel = not args.is_haemek

def get_slide_label(root, file, code_dict, size_factor, ind):
    slide = openslide.OpenSlide(os.path.join(root, file))
    label_im = slide.associated_images['label']
    n_barcodes = 0

    if size_factor != 1:
        w, h = label_im.size
        label_im = label_im.resize((int(w * size_factor), int(h * size_factor)), resample=Image.BICUBIC)


    all_codes = decode(label_im)


    for code in all_codes:
        barcode = code.data.decode('UTF-8')
        # try: #RanS 30.11.20, sometimes it finds garbage codes
        if 16 >= len(code.data) >= 10:  # RanS 1.12.20, barcodes are 10 to 16 characters long
            if is_carmel:
                barcode_adjust = str(int(barcode[0:4]) - 9788) + '-' + barcode[4:]
            else:  # haemek
                year = int(barcode[4]) + 14
                barcode_adjust = str(year) + '-' + barcode[5:]

            n_barcodes += 1
            #barcode_adjust = str(int(barcode[0:4]) - 9788) + '-' + barcode[4:]
            #if barcode_adjust[0] == '1' and barcode_adjust[2] == '-': #RanS 27.12.20, more sanity validation. needs to be regex realyy
            #if re.search("\d{2}-\d+/\d+/\d+/\w", barcode_adjust) != None:
            if re.search("\d{2}-\d+/\d+/\d+/\D{1}\Z", barcode_adjust) != None: #RanS 22.4.21
                #code_dict = {'SlideID': barcode_adjust, 'file': file, 'dir': root, 'unreadable': ''}
                code_dict[ind] = {'dir': root, 'file': file, 'SlideID': barcode_adjust,  'unreadable': ''}
                #df = df.append(code_dict, ignore_index=True)
        #code_dict = {'SlideID': '-1', 'file': file, 'dir': root}
        #df = df.append(code_dict, ignore_index=True)
    return code_dict, n_barcodes


slides_contain_barcodes = args.has_barcodes #if false, skip barcode search and print label image to file
save_slide_labels = args.save_labels #if false, skip printing barcode/label image
walk_dir = args.walk_dir

print('walk_dir = ' + walk_dir)
start_time = time.time()
prev_time = start_time

excel_file = os.path.join(walk_dir, '../../barcode_list.xlsx')
code_dict = OrderedDict()

if not os.path.isdir(os.path.join(walk_dir, 'unreadable_labels')):
    os.mkdir(os.path.join(walk_dir, 'unreadable_labels'))

ind = -1
for root, subdirs, files in os.walk(walk_dir):
    print('--\nfolder = ' + root)

    #for each mrxs file, open the slide and get the label
    if args.is_jpg:
        slide_files = [file for file in files if file[-4:] == '.jpg']
    else:
        slide_files = [file for file in files if file[-5:] == '.mrxs']
        slide_files.extend([file for file in files if file[-4:] == '.svs'])
        slide_files.extend([file for file in files if file[-5:] == '.tiff'])
        slide_files.extend([file for file in files if file.split('.')[-1] == 'isyntax']) # RanS 4.1.21
        slide_files.extend([file for file in files if file.split('.')[-1] == 'ndpi'])  # RanS 6.3.22

    for file in slide_files:
        ind += 1
        #if len(df.loc[(df['file'] == file) & (df['dir'] == root)]) > 0:
        #    continue

        print('processing image: ' + file)

        #temp RanS 14.12.20, this file is causing failure with exit code 40
        #if file == '3M04.mrxs':
        #if file == '2M06.mrxs' and root == r'\\gipnetappa\public\sgils\BCF scans\Carmel Slides\Batch_3\2020-12-03 box 6':
        #if file == '4M11.mrxs' and root == r'\\gipnetappa\public\sgils\BCF scans\Carmel Slides\Batch_6\2021-03-16 box 21':
        if (file == '6M08.mrxs' and root == r'/mnt/gipmed_new/Data/Breast/Carmel/Batch_9/2021-06-20 box 47') or \
            (file == '48-36.mrxs' and root == r'/mnt/gipmed_new/Data/Breast/Carmel/Batch_9/2021-06-27 box 48') or \
            (file == '10M16.mrxs' and root == r'/mnt/gipmed_new/Data/Breast/Carmel/Batch_9/2021-07-25 box 54') or \
            (file == '20-7231_1_1_e.mrxs') or (file == '20-8642_1_1_e.mrxs'):
            #code_dict_0 = {'SlideID': '', 'file': file, 'dir': root, 'unreadable': 'skipped'}
            #df = df.append(code_dict_0, ignore_index=True)
            code_dict[ind] = {'dir': root, 'file': file, 'SlideID': '', 'unreadable': 'skipped'}
            continue

        size_factor = 0.1
        n_barcodes = 0

        if slides_contain_barcodes and save_slide_labels:
            try:
                while n_barcodes == 0 and size_factor <= 1:
                    #df, n_barcodes = get_slide_label(root, file, df, size_factor=size_factor)
                    code_dict, n_barcodes = get_slide_label(root, file, code_dict, size_factor=size_factor, ind=ind)
                    size_factor *= 2
                    if size_factor > 0.5 and size_factor < 1:
                        size_factor = 1
            except:
                print('failed during get_slide_label')
        if n_barcodes > 1:
            print('more than one barcode found!')
        if n_barcodes == 0:
            if save_slide_labels:
                print('no barcodes found! saving label to image')
                try:
                    slide = openslide.OpenSlide(os.path.join(root, file))
                    label_im = slide.associated_images['label']
                    slide.close()
                    plt.imshow(label_im)
                    label_name = str(ind).zfill(4) + root.replace('/', '_') + '_' + os.path.splitext(file)[0] + '.png'
                    plt.savefig(os.path.join(walk_dir, 'unreadable_labels', label_name))
                    #code_dict_0 = {'SlideID': '', 'file': file, 'dir': root, 'unreadable': 'True'}
                    code_dict[ind] = {'dir': root, 'file': file, 'SlideID': '', 'unreadable': 'True'}
                except:
                    #code_dict_0 = {'SlideID': '', 'file': file, 'dir': root, 'unreadable': 'cannot open slide'}
                    code_dict[ind] = {'dir': root, 'file': file, 'SlideID': '', 'unreadable': 'cannot open slide'}
            else:
                #code_dict_0 = {'SlideID': '', 'file': file, 'dir': root}
                code_dict[ind] = {'dir': root, 'file': file, 'SlideID': ''}
            #df = df.append(code_dict_0, ignore_index=True)
        if n_barcodes == 1:
            print('working size factor is ', str(size_factor/2))
        if ind % 100 == 0: #save every 100 slides
            df = pd.DataFrame.from_dict(code_dict, "index")
            df.to_excel(excel_file)  # save after every image
        image_time = time.time()
        print('processing time: ' + str(image_time - prev_time) + ' sec')
        prev_time = image_time

df = pd.DataFrame.from_dict(code_dict, "index")
df.to_excel(excel_file)
print('Finished, total time: ' + str(time.time() - start_time) + ' sec')