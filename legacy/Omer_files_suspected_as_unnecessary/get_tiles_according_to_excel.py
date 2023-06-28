import pandas as pd

import Dataset_Maker.dataset_utils
import utils
import os
import openslide
import sys
from tqdm import tqdm
import argparse
from pathlib import Path

"""
In order to extract tiles in their original boundaries the flag -o should be used. Pay attention that those tiles will
be resized according to the ORIGINAL_MAGNIFICATION which should not be changed from 10 and the args.desired_magnification
which should be 40 for maximum resolution
"""

parser = argparse.ArgumentParser(description='Extract tiles from excel')
parser.add_argument('-o', dest='original', action='store_true', help='Extract original size  ?')
parser.add_argument('-mag', dest='desired_magnification', type=int, default=40, help='Desired Magnification for tiles')
parser.add_argument('-png', dest='png', action='store_true', help='Save as .png ?')
parser.add_argument('--file', dest='file', type=str, default='low_grade_patches_to_extract_Batch9', help='file name to use')
args = parser.parse_args()

TILE_SIZE = 256
ORIGINAL_MAGNIFICATION = 10

if sys.platform == 'darwin':
    highest_DF = pd.read_excel(r'/Users/wasserman/Developer/WSI_MIL/Data For Gil/' + args.file +'.xlsx')
    #lowest_DF = pd.read_excel(r'/Users/wasserman/Developer/WSI_MIL/Data For Gil/patches_to_extract_neg.xlsx')
elif sys.platform == 'linux':
    highest_DF = pd.read_excel(r'/home/womer/project/Data For Gil/' + args.file + '.xlsx')
    #lowest_DF = pd.read_excel(r'/home/womer/project/Data For Gil/patches_to_extract_lowest_normalized.xlsx')

highest_slide_filenames = list(highest_DF['SlideName'])
highest_tile_indices = list(highest_DF['TileIdx'])
highest_tile_locations = []
highest_image_outputnames = list(highest_DF['OutputName'])
for tile_idx in range(len(highest_DF)):
    highest_tile_locations.append((highest_DF['TileLocation1'][tile_idx], highest_DF['TileLocation2'][tile_idx]))

'''lowest_slide_filenames = list(lowest_DF['SlideName'])
lowest_tile_indices = list(lowest_DF['TileIdx'])
lowest_tile_locations = []
lowest_image_outputnames = list(lowest_DF['OutputName'])
for tile_idx in range(len(lowest_DF)):
    lowest_tile_locations.append([lowest_DF['TileLocation1'][tile_idx], lowest_DF['TileLocation2'][tile_idx]])'''

# Extracting the tiles:
# open slides_data.xlsx file:

slide_data_file_carmel = r'/home/womer/project/All Data/Ran_Features/Grid_data/slides_data_CARMEL_ALL.xlsx' if sys.platform == 'linux' else r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/Grids_data/slides_data_CARMEL_ALL.xlsx'
slide_data_file_carmel_9_11 = r'/home/womer/project/All Data/Ran_Features/Grid_data/slides_data_CARMEL_9_11.xlsx' if sys.platform == 'linux' else r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/Grids_data/slides_data_CARMEL_9_11.xlsx'

slides_meta_data_DF_carmel = pd.read_excel(slide_data_file_carmel)
slides_meta_data_DF_carmel_9_11 = pd.read_excel(slide_data_file_carmel_9_11)
slides_meta_data_DF = pd.concat((slides_meta_data_DF_carmel, slides_meta_data_DF_carmel_9_11))
slides_meta_data_DF.set_index('file', inplace=True)
batches = {'Highest': [highest_slide_filenames, highest_tile_indices, highest_tile_locations, highest_image_outputnames]}
'''batches = {'Highest': [highest_slide_filenames, highest_tile_indices, highest_tile_locations, highest_image_outputnames],
           'Lowest': [lowest_slide_filenames, lowest_tile_indices, lowest_tile_locations, lowest_image_outputnames]}'''

if args.file.split('_')[-1] in ['Batch9', 'Batch10', 'Batch11']:
    Dataset = 'Carmel 9-11'
else:
    Dataset = 'CARMEL'

for key in batches.keys():
    slide_filenames = batches[key][0]
    tile_indices = batches[key][1]
    tile_locations = batches[key][2]
    outputnames = batches[key][3]

    for file_idx in tqdm(range(len(tile_indices))):
        if sys.platform == 'darwin':
            dir_dict = Dataset_Maker.dataset_utils.get_datasets_dir_dict(Dataset=Dataset)
            image_file = os.path.join(dir_dict['CARMEL'], slide_filenames[file_idx])
        elif sys.platform == 'linux':
            dir_dict = Dataset_Maker.dataset_utils.get_datasets_dir_dict(Dataset=Dataset)
            file_id = slides_meta_data_DF.loc[slide_filenames[file_idx]]['id']
            image_file = os.path.join(dir_dict[file_id], slide_filenames[file_idx])

        tile_location = tile_locations[file_idx]
        tile_index_in_xl = tile_indices[file_idx]


        slide = openslide.open_slide(image_file)
        # Compute level to extract tile from:
        slide_level_0_magnification = slides_meta_data_DF.loc[slide_filenames[file_idx]]['Manipulated Objective Power']

        # Choosing to extract tiles in original size will extract tiles that originated from magnification 10
        # (ORIGINAL_MAGNIFICATION) at size 256^2 pixels (TILE_SIZE) and will change the tile size according to the desired_magnification
        # such the the tile margins will remain the same
        if args.original:
            magnification_ratio = args.desired_magnification // ORIGINAL_MAGNIFICATION
            #desired_magnification = slide_level_0_magnification
            tile_size = TILE_SIZE * magnification_ratio
        else:
            #desired_magnification = args.desired_magnification
            tile_size = TILE_SIZE

        best_slide_level, adjusted_tile_size, level_0_tile_size = utils.get_optimal_slide_level(slide=slide,
                                                                                                magnification=slide_level_0_magnification,
                                                                                                desired_mag=args.desired_magnification,
                                                                                                tile_size=tile_size)

        # Get the tiles:
        image_tiles, _, _ = utils._get_tiles(slide=slide,
                                             locations=[tile_location],
                                             tile_size_level_0=level_0_tile_size,
                                             adjusted_tile_sz=adjusted_tile_size,
                                             output_tile_sz=tile_size,
                                             best_slide_level=best_slide_level
                                             )

        # Save tile:
        if not os.path.isdir(os.path.join('Data For Gil', args.file)):
            Path(os.path.join('Data For Gil', args.file)).mkdir(parents=True)

        tile_number = (8 - len(str(tile_index_in_xl))) * '0' + str(tile_index_in_xl)
        tile_filename = os.path.join('Data For Gil', args.file, outputnames[file_idx])  #  os.path.join('Data For Gil', tile_number)
        tile_filename_extension = '.png' if args.png else '.png'
        image_tiles[0].save(tile_filename + tile_filename_extension)

print('Done')