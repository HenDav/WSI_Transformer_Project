import os
import pandas as pd
from shutil import copyfile

'''
#temp - get file list
import csv
main_dir = r'C:\ran_data\TCGA_segmentation_v3\SegImages'
out_dir = r'C:\ran_data\TCGA_segmentation_v3\SegImages_numbered'
mylist = os.listdir(os.path.join(main_dir))
ind = 0
for file in os.listdir(os.path.join(main_dir)):
    ind += 1
    copyfile(os.path.join(main_dir, file), os.path.join(out_dir, str(ind).zfill(4) + '_' + file))


import numpy as np
np.savetxt(r'C:\ran_data\TCGA_segmentation_v3\file_list1.csv', mylist, delimiter=",", fmt='%s')

with open(r'C:\ran_data\TCGA_segmentation_v3\file_list1.csv', 'wb') as f1:
    wr = csv.writer(f1, quoting=csv.QUOTE_ALL)
    wr.writerow(mylist)'''


if __name__ == '__main__':
    main_dir = r'C:\ran_data\TCGA_segmentation_v3\SegImages'
    DX_dir = r'C:\ran_data\TCGA_segmentation_v3\SegImages_DX'
    others_dir = r'C:\ran_data\TCGA_segmentation_v3\SegImages_TS_BS_MS'

    legend_file = r'C:\ran_data\gdc_manifest.2020-06-09_TCGA_legend.csv'
    data = pd.read_csv(legend_file)
    ind = 0
    for file in os.listdir(os.path.join(main_dir)):
        ind += 1
        print(file)
        slide_id = file[:-13]
        #slide_type = data.loc[data['id'] == slide_id]['slidetype'].item()
        slide_type = data.loc[data['filename2'] == slide_id]['slidetype'].item()
        if slide_type=='DX':
            # copy to DX folder
            copyfile(os.path.join(main_dir, file), os.path.join(DX_dir, str(ind).zfill(4) + '_' + file))
        else:
            # copy to other folder
            copyfile(os.path.join(main_dir, file), os.path.join(others_dir, str(ind).zfill(4) + '_' + file))
    print('finished')

