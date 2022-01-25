import os
import argparse

parser = argparse.ArgumentParser(description='WSI_REG Slide inference')
parser.add_argument('-d', '--indir', type=str, default='/mnt/gipmed_new/Data/Breast/Sheba/isyntax/batch3_originals', help='input dir')
parser.add_argument('-o', '--outdir', type=str, default='/mnt/gipmed_new/Data/Breast/Sheba/isyntax/batch3_converted', help='output dir')
args = parser.parse_args()

counter = 0
batch_text = '@ECHO off\n\n'
for root, subdirs, files in os.walk(args.indir):
    slide_files = [file for file in files if file[-8:] == '.isyntax']
    for file in slide_files:
        print('file:', file)
        full_fn = os.path.join(root, file)
        out_full_fn = os.path.join(args.outdir, file[:-8] + '.tiff')
        batch_text += 'ECHO Converting ' + str(counter) + '\n'
        batch_text += 'ConvertToTiff --inputFile ' + full_fn + ' --outputFile ' + out_full_fn + ' --scanFactor 20\n'
        counter += 1

batch_text += '\nECHO Done!'
#fit to windown format
batch_text = batch_text.replace('/', '\\')
batch_text = batch_text.replace('mnt', r'\gipnetappa')

#text_file = open("/home/rschley/code/WSI_MIL/general_try4/python_batch4_050121.bat", "w")
text_file = open(os.path.join(args.outdir, "python_batch5_250122.bat"), "w")
text_file.write(batch_text)
text_file.close()
print('Finished')