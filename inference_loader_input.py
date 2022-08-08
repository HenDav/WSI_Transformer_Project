import os
import re, glob
import sys

inference_files = {}

exp = 876
fold = '5'
target = 'onco_score_all'
dataset = 'SHEBA'
subdir = ''
is_other = False
csv_epoch = 2000

patientless_list = ['TCGA_LUNG', 'HEROHE']
if dataset in patientless_list or subdir in patientless_list:
    patient_level = False
else:
    patient_level = True
save_csv = True

if sys.platform == 'win32':
    if is_other:
        inference_dir = os.path.join(r'C:\Pathnet_results\MIL_general_try4', dataset + '_runs', 'other', 'exp' + str(exp), 'Inference')
    else:
        inference_dir = os.path.join(r'C:\Pathnet_results\MIL_general_try4', dataset + '_runs', target, 'exp' + str(exp), 'Inference')
elif sys.platform == 'darwin':
    inference_dir = glob.glob(os.path.join(r'/Users/wasserman/Developer/WSI_MIL/runs', 'Exp_' + str(exp) + '*'))[0] + '/Inference/'
    target = inference_dir.split('/')[6].split('-')[1]

    if target == 'Survival_Time_Cox':
        target = 'survival'


inference_dir = os.path.join(inference_dir, subdir)
# auto find epochs
file_list = glob.glob(os.path.join(inference_dir, '*.data'))
epochs = [int(re.findall(r"\bModel_Epoch_+\d+\b", os.path.basename(fn))[0][12:]) for fn in file_list]
epochs.sort()

key_list = [''.join((target, '_fold', str(fold), '_exp', str(exp), '_epoch', str(epoch), '_test_500')) for epoch in epochs]
val_list = [''.join(('Model_Epoch_', str(epoch), '-Folds_[', str(fold), ']_', target, '-Tiles_500.data')) for epoch in epochs]

# manual
temp = False
if temp:
    val_list = ['Model_Epoch_1000-Folds_[1, 2, 3, 4, 5]_ER+PR+Her2-Tiles_5.data']
    key_list = ['temp']
    inference_dir = r'C:\Users\User\Dropbox\Technion work 2020\Code\WSI_MIL\WSI_MIL\runs\Exp_621-ER+PR+Her2-TestFold_1\Inference'

inference_files = dict(zip(key_list, val_list))

inference_name = target + '_fold' + str(fold) + '_exp' + str(exp)
