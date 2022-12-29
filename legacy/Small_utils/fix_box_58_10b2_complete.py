import pandas as pd
import os
import numpy as np
import shutil

dn = r'C:\ran_data\Carmel_Slides_examples\boxes58_10b2 fix'
dn9 = r'/mnt/gipmed_new/Data/Breast/Carmel/Batch_9/CARMEL9'
dn10 = r'/mnt/gipmed_new/Data/Breast/Carmel/Batch_10/CARMEL10'
dat_file = r'/mnt/gipmed_new/Data/Breast/Carmel/the_correct_boxes_RanS 310821 fixed.xlsx'

slides_data_file9 = r'Slides_data_CARMEL9.xlsx'
slides_data_file10 = r'Slides_data_CARMEL10.xlsx'

s9 = pd.read_excel(os.path.join(dn9, slides_data_file9))
s10 = pd.read_excel(os.path.join(dn10, slides_data_file10))
dat = pd.read_excel(os.path.join(dat_file))
currently_in_9 = []
currently_in_10 = []
for row in dat.iterrows():
    print(row)
    #row[1]['currently_in'] = 50
    current_box = row[1]['current box']
    correct_box = row[1]['correct box']
    if current_box != correct_box:
        #move to correct box
        slide_name = row[1]['slide_rename']
        if current_box == 10:
            shutil.move(os.path.join(dn10, slide_name + '.mrxs'), os.path.join(dn10, 'move_to_batch9', slide_name + '.mrxs'))
            shutil.move(os.path.join(dn10, slide_name), os.path.join(dn10, 'move_to_batch9', slide_name))
        elif current_box == 9:
            shutil.move(os.path.join(dn9, slide_name + '.mrxs'), os.path.join(dn9, 'move_to_batch10', slide_name + '.mrxs'))
            shutil.move(os.path.join(dn9, slide_name), os.path.join(dn9, 'move_to_batch10', slide_name))

print('finished')