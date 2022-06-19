import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='WSI_REG Slide inference')
parser.add_argument('--in_dir', type=str, default=r'C:\ran_data\BoneMarrow\new_slides_041021', help='input dir')
parser.add_argument('--out_dir', type=str, default='LEUKEMIA', help='output dir')

parser.add_argument('--is_mrxs', action='store_true', help='is mrxs slide') #RanS 8.2.21

args = parser.parse_args()
in_dir = args.in_dir
out_dir = os.path.join(args.in_dir, args.out_dir)
is_mrxs = args.is_mrxs

if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

excel_file = os.path.join(in_dir, '../barcode_list.xlsx')
if os.path.isfile(excel_file):
    barcode_list = pd.read_excel(excel_file, engine='openpyxl')
else:
    raise IOError('cannot find barcode file!')

for row in barcode_list.iterrows():
    print(row)
    fn = os.path.join(row[1]['dir'], row[1]['file'])
    try:
        if is_mrxs:
            dn = fn[:-5]
            #if slide file and folder exist:
            if os.path.isfile(fn) and os.path.isdir(dn) and row[1]['slide rename'] != -1:
                os.rename(fn, os.path.join(out_dir, str(row[1]['slide rename']) + '.mrxs'))
                os.rename(dn, os.path.join(out_dir, str(row[1]['slide rename'])))
                print("Source path renamed to destination path successfully.")

        else: # svs and other files
            if os.path.isfile(fn) and row[1]['slide rename'] != -1:
                os.rename(fn, os.path.join(out_dir, str(row[1]['slide rename'])))
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
