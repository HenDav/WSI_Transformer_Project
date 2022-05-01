import os
import pandas as pd
import argparse

#delete slides (move to a different folder) according to file
#assume we already have a "deleted slides" folder with an slides_to_delete.xlsx file in it

parser = argparse.ArgumentParser(description='WSI_REG Slide inference')
parser.add_argument('--in_dir', type=str, default=r'/mnt/gipmed_new/Data/Breast/Carmel/Benign/Batch_3/BENIGN3', help='input dir')

#parser.add_argument('--is_mrxs', action='store_true', help='is mrxs slide') #RanS 8.2.21

args = parser.parse_args()
in_dir = args.in_dir
#is_mrxs = args.is_mrxs

out_dir = os.path.join(os.path.dirname(in_dir), 'deleted slides')

excel_file = os.path.join(out_dir, 'slides_to_delete.xlsx')
if os.path.isfile(excel_file):
    slide_list = pd.read_excel(excel_file, engine='openpyxl')
else:
    raise IOError('cannot find barcode file!')

for row in slide_list.iterrows():
    #print(row)
    #fn = os.path.join(row[1]['dir'], row[1]['file'])
    fn = os.path.join(in_dir, row[1]['file'])
    print(fn)
    try:
        if os.path.splitext(fn)[-1] == '.mrxs':
            full_dn = fn[:-5]
            dn = row[1]['file'][:-5]
            #if slide file and folder exist:
            if os.path.isfile(fn) and os.path.isdir(full_dn):
                os.rename(fn, os.path.join(out_dir, row[1]['file']))
                os.rename(full_dn, os.path.join(out_dir, dn))
                print("Source path renamed to destination path successfully.")

        else: # svs and other files
            if os.path.isfile(fn):
                os.rename(fn, os.path.join(out_dir, row[1]['file']))
                print("Source path renamed to destination path successfully.")

    # If Source is a file
    # but destination is a directory
    except IsADirectoryError:
        print("Source is a file but destination is a directory")
        # If source is a directory
    # but destination is a file
    except NotADirectoryError:
        print("Source is a directory but destination is a file")
        # For permission related errors
    except PermissionError:
        print("Operation not permitted")
        # For other errors
    except OSError as error:
        print(error)
print('finished')
