import os
import openslide
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
from shutil import copyfile
import argparse
import pandas as pd
import pickle
from Dataset_Maker import dataset_utils
import sys

curr_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(curr_dir)
sys.path.append(parent_dir)
from utils import _choose_data

rewrite_figs = True


def create_slide_inspection_folder(in_dir, out_dir, desired_mag, grid_only=False, grid_path_name='', thumbs_only=False):
    grid_image_path = get_grid_image_path(in_dir, grid_path_name, thumbs_only)

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    slide_files_svs = glob.glob(os.path.join(in_dir, '*.svs'))
    slide_files_ndpi = glob.glob(os.path.join(in_dir, '*.ndpi'))
    slide_files_mrxs = glob.glob(os.path.join(in_dir, '*.mrxs'))
    slide_files_tiff = glob.glob(os.path.join(in_dir, '*.tiff'))
    slide_files_tif = glob.glob(os.path.join(in_dir, '*.tif'))
    slides = np.sort(slide_files_svs + slide_files_ndpi + slide_files_mrxs + slide_files_tiff + slide_files_tif)

    dataset = os.path.basename(in_dir)
    slide_meta_data_file = os.path.join(in_dir, 'slides_data_' + dataset + '.xlsx')
    slide_meta_data_DF = dataset_utils.open_excel_file(slide_meta_data_file)
    if not thumbs_only:
        grid_meta_data_file = os.path.join(in_dir, 'Grids_' + str(desired_mag), 'Grid_data.xlsx')
        grid_meta_data_DF = dataset_utils.open_excel_file(grid_meta_data_file)
        meta_data_DF = pd.merge(slide_meta_data_DF, grid_meta_data_DF, on="file")
    else:
        meta_data_DF = slide_meta_data_DF

    ind = 0
    fn_list = []

    for _, file in enumerate(tqdm(slides)):
        fn_full = os.path.basename(file)
        fn = os.path.splitext(fn_full)[0]
        out_path = os.path.join(out_dir, fn + '.jpg')

        need_to_process_slide = rewrite_figs or not os.path.isfile(out_path)
        if need_to_process_slide:
            if grid_only or thumbs_only:
                slide_mag = 0
                n_legit_tiles = 0
            else:
                try:
                    slide_mag = meta_data_DF.loc[meta_data_DF['file'] == fn_full, 'Manipulated Objective Power'].item()
                    n_legit_tiles = meta_data_DF.loc[meta_data_DF['file'] == fn_full,
                                                     'Legitimate tiles - 256 compatible @ X' + str(desired_mag)].values[0]
                except:
                    print('fn:', fn, ' had problem with slides data (multiple identical filenames?)')
                    n_legit_tiles = -1

            fn = slide_2_image(in_dir, out_dir, grid_image_path, file, ind, slide_mag, n_legit_tiles, desired_mag,
                               grid_only, thumbs_only)
            if fn != -1:
                fn_list.append(fn)
        ind += 1

    fn_list_df = pd.DataFrame(fn_list)
    fn_list_df.to_excel(os.path.join(out_dir, 'slide_review_list.xlsx'))
    print('finished')


def slide_2_image(in_dir, out_dir, grid_image_path, slide_file, ind, slide_mag, n_legit_tiles, desired_mag,
                  grid_only, thumbs_only):
    fn = os.path.splitext(os.path.basename(slide_file))[0]
    success_flag = True
    get_random_patch_images = (not grid_only) and (not thumbs_only)
    if get_random_patch_images:
        grid_file = os.path.join(in_dir, 'Grids_' + str(desired_mag), fn + '--tlsz256' + '.data')
        if not os.path.isfile(grid_file):
            print('no grid file')
            return -1
        # get random patches
        fig = plt.figure()
        fig.set_size_inches(32, 18)
        grid_shape = (4, 8)
        n_patches = int(np.prod(grid_shape))
        grid = ImageGrid(fig, 111, nrows_ncols=grid_shape, axes_pad=0)
        patch_size = 256
        n_patches = np.minimum(n_patches, n_legit_tiles)
        slide = openslide.open_slide(slide_file)
        with open(grid_file, 'rb') as filehandle:
            grid_list = pickle.load(filehandle)
        if n_patches == -1:
            print('no valid patches found for slide ', fn)
            success_flag = False
        tiles, time_list, _, _ = _choose_data(grid_list, slide, n_patches, slide_mag, patch_size, False, desired_mag, False,
                                              False)
        for ii in range(n_patches):
            grid[ii].imshow(tiles[ii])
            grid[ii].set_yticklabels([])
            grid[ii].set_xticklabels([])
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, str(ind).zfill(4) + '_2_patches_' + fn + '.jpg'))
        plt.close()
    # thumb image
    if os.path.isfile(os.path.join(in_dir, 'SegData', 'Thumbs', fn + '_thumb.jpg')):
        copyfile(os.path.join(in_dir, 'SegData', 'Thumbs', fn + '_thumb.jpg'),
                 os.path.join(out_dir, str(ind).zfill(4) + '_0_thumb_' + fn + '.jpg'))
    # elif os.path.isfile(os.path.join(in_dir, 'SegData', 'Thumbs', fn + '_thumb.png')):  # old format
    #     copyfile(os.path.join(in_dir, 'SegData', 'Thumbs', fn + '_thumb.png'),
    #              os.path.join(out_dir, str(ind).zfill(4) + '_0_thumb_' + fn + '.png'))
    else:
        print('no thumb image found for slide ' + fn)
        success_flag = False
    # grid image
    if not thumbs_only:
        if os.path.isfile(os.path.join(grid_image_path, fn + '_GridImage.jpg')):
            copyfile(os.path.join(grid_image_path, fn + '_GridImage.jpg'),
                     os.path.join(out_dir, str(ind).zfill(4) + '_1_GridImage_' + fn + '.jpg'))
        else:
            print('no grid image found for slide ' + fn)
            success_flag = False
    if success_flag:
        return fn
    else:
        return -1


def get_grid_image_path(in_dir, grid_path_name, thumbs_only):
    if thumbs_only:
        return -1
    if grid_path_name != '':
        grid_image_path = os.path.join(in_dir, 'SegData', 'GridImages_' + grid_path_name)
    else:
        grid_image_paths = glob.glob(os.path.join(in_dir, 'SegData', 'GridImages*'))
        assert len(grid_image_paths) > 0, 'no GridImages Folder!'
        assert len(grid_image_paths) == 1, 'more than one GridImages Folder! define a specific folder'
        grid_image_path = grid_image_paths[0]
    return grid_image_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Slide inspector')
    parser.add_argument('--in_dir', default=r'/mnt/gipnetapp_public/sgils/BCF scans/Carmel Slides/Batch_6/CARMEL6',
                        type=str, help='input dir')
    parser.add_argument('--out_dir', default=r'/mnt/gipnetapp_public/sgils/BCF scans/Carmel Slides/Batch_6/thumbs',
                        type=str, help='output dir')
    parser.add_argument('--mag', type=int, default=10, help='desired magnification of patches')
    parser.add_argument('--grid_only', action='store_true', help='plot grid images only')
    parser.add_argument('--grid_path_name', default='', type=str, help='extension of grid_images path')
    parser.add_argument('--thumbs_only', action='store_true', help='create only thumbnails')
    args = parser.parse_args()
    create_slide_inspection_folder(in_dir=args.in_dir,
                                   out_dir=args.out_dir,
                                   desired_mag=args.mag,
                                   grid_only=args.grid_only,
                                   grid_path_name=args.grid_path_name,
                                   thumbs_only=args.thumbs_only)
