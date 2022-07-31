import logging
import os
import pickle
import sys
import time
from glob import glob
from random import sample, choices
from typing import List

import numpy as np
import openslide

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from Dataset_Maker.dataset_utils import get_datasets_dir_dict
from datasets import WSI_Master_Dataset, WSI_Master_Dataset_Survival
from transformations import define_transformations
from utils import assert_dataset_target, map_original_grid_list_to_equiv_grid_list, chunks, get_optimal_slide_level, \
    get_label, _get_tiles, _choose_data, num_2_bool


class WSI_Master_Dataset_Ver_2(Dataset):
    def __init__(self,
                 dataset: str = 'TCGA',
                 tile_size: int = 256,
                 bag_size: int = 50,
                 target_kind: str = 'ER',
                 test_fold: int = 1,
                 infer_folds: List = [None],
                 train: bool = True,
                 print_timing: bool = False,
                 transform_type: str = 'flip',
                 DX: bool = False,
                 get_images: bool = False,
                 train_type: str = 'MASTER',
                 color_param: float = 0.1,
                 n_tiles: int = 10,
                 test_time_augmentation: bool = False,
                 desired_slide_magnification: int = 10,
                 slide_repetitions: int = 1,
                 loan: bool = False,
                 er_eq_pr: bool = False,
                 slide_per_block: bool = False,
                 balanced_dataset: bool = False,
                 RAM_saver: bool = False
                 ):

        self._set_input_parameters(input_parameters=locals())
        self._exception_check_for_input_arguments()
        self._set_processed_parameters()
        self._create_metadata_DataFrame()

        # Make sure that there are no exceptions needed to be thrown concerning the attributes
        # filter the whole database w.r.t it's attributes and remain with the legitimate samples.
        # make those samples ready for use with the __getitem__() method.

        pass

    def _set_input_parameters(self, input_parameters):
        input_parameters.pop('self')
        self.input_parameters = input_parameters

    def _is_multi_target(self):
        return len(self.input_parameters['target_kind'].split('+')) > 1

    def _set_multi_target(self):
        if self._is_multi_target():
            self.input_parameters['target_kind'] = self.input_parameters['target_kind'].split('+')
            self.processed_parameters['N_targets'] = 2  # currently support only two targets!
            self.processed_parameters['multi_target'] = True
        else:
            self.processed_parameters['multi_target'] = False

    def _set_processed_parameters(self):
        self.processed_parameters = {}
        self._set_multi_target()

    def _create_metadata_DataFrame(self):
        self.dir_dict = get_datasets_dir_dict(Dataset=self.input_parameters['dataset'])
        logging.info('Slide Data will be taken from these locations:')
        logging.info(self.dir_dict)
        locations_list = []

        for _, key in enumerate(self.dir_dict):
            locations_list.append(self.dir_dict[key])

            slide_meta_data_file = os.path.join(self.dir_dict[key], 'slides_data_' + key + '.xlsx')
            grid_meta_data_file = os.path.join(self.dir_dict[key],
                                               'Grids_' + str(self.desired_magnification),
                                               'Grid_data.xlsx')

            slide_meta_data_DF = pd.read_excel(slide_meta_data_file)
            grid_meta_data_DF = pd.read_excel(grid_meta_data_file)
            meta_data_DF = pd.DataFrame({**slide_meta_data_DF.set_index('file').to_dict(),
                                         **grid_meta_data_DF.set_index('file').to_dict()})

            self.meta_data_DF = meta_data_DF if not hasattr(self, 'meta_data_DF') else self.meta_data_DF.append(
                meta_data_DF)
        self.meta_data_DF.reset_index(inplace=True)
        self.meta_data_DF.rename(columns={'index': 'file'}, inplace=True)

    def _exception_check_for_input_arguments(self):
        assert_dataset_target(DataSet, target_kind)

    def __len__(self):
        return -1

    def __getitem__(self, item):
        return -1


class WSI_MILdataset(WSI_Master_Dataset):
    def __init__(self,
                 DataSet: str = 'TCGA',
                 tile_size: int = 256,
                 bag_size: int = 10,
                 target_kind: str = 'ER',
                 test_fold: int = 1,
                 train: bool = True,
                 print_timing: bool = False,
                 transform_type: str = 'flip',
                 DX: bool = False,
                 get_images: bool = False,
                 color_param: float = 0.1,
                 test_time_augmentation: bool = False,
                 desired_slide_magnification: int = 20,
                 slide_repetitions: int = 1
                 ):
        super(WSI_MILdataset, self).__init__(DataSet=DataSet,
                                             tile_size=tile_size,
                                             bag_size=bag_size,
                                             target_kind=target_kind,
                                             test_fold=test_fold,
                                             train=train,
                                             print_timing=print_timing,
                                             transform_type=transform_type,
                                             DX=DX,
                                             get_images=get_images,
                                             train_type='MIL',
                                             color_param=color_param,
                                             test_time_augmentation=test_time_augmentation,
                                             desired_slide_magnification=desired_slide_magnification,
                                             slide_repetitions=slide_repetitions)

        logging.info(
            'Initiation of WSI({}) {} {} DataSet for {} is Complete. Magnification is X{}, {} Slides, Tiles of size {}^2. {} tiles in a bag, {} Transform. TestSet is fold #{}. DX is {}'
                .format(self.train_type,
                        'Train' if self.train else 'Test',
                        self.DataSet,
                        self.target_kind,
                        self.desired_magnification,
                        self.real_length,
                        self.tile_size,
                        self.bag_size,
                        'Without' if transform_type == 'none' else 'With',
                        self.test_fold,
                        'ON' if self.DX else 'OFF'))


class Full_Slide_Inference_Dataset(WSI_Master_Dataset):
    def __init__(self,
                 DataSet: str = 'TCGA',
                 tile_size: int = 256,
                 tiles_per_iter: int = 500,
                 target_kind: str = 'ER',
                 folds: List = [1],
                 dx: bool = False,
                 desired_slide_magnification: int = 10,
                 num_background_tiles: int = 0
                 ):
        super(Full_Slide_Inference_Dataset, self).__init__(DataSet=DataSet,
                                                           tile_size=tile_size,
                                                           bag_size=None,
                                                           target_kind=target_kind,
                                                           test_fold=1,
                                                           infer_folds=folds,
                                                           train=True,
                                                           print_timing=False,
                                                           transform_type='none',
                                                           DX=dx,
                                                           get_images=False,
                                                           train_type='Infer',
                                                           desired_slide_magnification=desired_slide_magnification)

        self.tiles_per_iter = tiles_per_iter
        self.folds = folds
        self.magnification = []
        self.num_tiles = []
        self.slide_grids = []
        self.equivalent_grid = []
        self.equivalent_grid_size = []
        self.is_tissue_tiles = []
        self.is_last_batch = []

        ind = 0
        for _, slide_num in enumerate(self.valid_slide_indices):
            if (self.DX and self.all_is_DX_cut[slide_num]) or not self.DX:
                # Recreate the basic slide grids:
                height = int(self.meta_data_DF.loc[slide_num, 'Height'])
                width = int(self.meta_data_DF.loc[slide_num, 'Width'])
                objective_power = self.meta_data_DF.loc[slide_num, 'Manipulated Objective Power']

                adjusted_tile_size_at_level_0 = int(
                    self.tile_size * (int(objective_power) / self.desired_magnification))
                equivalent_rows = int(np.ceil(height / adjusted_tile_size_at_level_0))
                equivalent_cols = int(np.ceil(width / adjusted_tile_size_at_level_0))
                basic_grid = [(row, col) for row in range(0, height, adjusted_tile_size_at_level_0) for col in
                              range(0, width, adjusted_tile_size_at_level_0)]
                equivalent_grid_dimensions = (equivalent_rows, equivalent_cols)
                self.equivalent_grid_size.append(equivalent_grid_dimensions)

                if len(basic_grid) != self.meta_data_DF.loc[
                    slide_num, 'Total tiles - ' + str(self.tile_size) + ' compatible @ X' + str(
                        self.desired_magnification)].item():
                    raise Exception('Total tile num do not fit')

                self.magnification.extend([self.all_magnifications[slide_num]])

                basic_file_name = '.'.join(self.image_file_names[ind].split('.')[:-1])
                grid_file = os.path.join(self.image_path_names[ind], 'Grids_' + str(self.desired_magnification),
                                         basic_file_name + '--tlsz' + str(self.tile_size) + '.data')

                # which_patches = sample(range(int(self.tissue_tiles[ind])), self.num_tiles[-1])

                with open(grid_file, 'rb') as filehandle:
                    tissue_grid_list = pickle.load(filehandle)
                # Compute wich tiles are background tiles and pick "num_background_tiles" from them.
                non_tissue_grid_list = list(set(basic_grid) - set(tissue_grid_list))
                selected_non_tissue_grid_list = sample(non_tissue_grid_list, num_background_tiles)
                # Combine the list of selected non tissue tiles with the list of all tiles:
                combined_grid_list = selected_non_tissue_grid_list + tissue_grid_list
                combined_equivalent_grid_list = map_original_grid_list_to_equiv_grid_list(adjusted_tile_size_at_level_0,
                                                                                          combined_grid_list)
                # We'll also create a list that says which tiles are tissue tiles:
                is_tissue_tile = [False] * len(selected_non_tissue_grid_list) + [True] * len(tissue_grid_list)

                self.num_tiles.append(len(combined_grid_list))

                chosen_locations_chunks = chunks(combined_grid_list, self.tiles_per_iter)
                self.slide_grids.extend(chosen_locations_chunks)

                chosen_locations_equivalent_grid_chunks = chunks(combined_equivalent_grid_list, self.tiles_per_iter)
                self.equivalent_grid.extend(chosen_locations_equivalent_grid_chunks)

                is_tissue_tile_chunks = chunks(is_tissue_tile, self.tiles_per_iter)
                self.is_tissue_tiles.extend(is_tissue_tile_chunks)
                self.is_last_batch.extend([False] * (len(chosen_locations_chunks) - 1))
                self.is_last_batch.append(True)

                ind += 1

        # The following properties will be used in the __getitem__ function
        # self.tiles_to_go = None
        # self.new_slide = True
        self.slide_num = -1
        self.current_file = None
        print('Initiation of WSI INFERENCE for {} DataSet and {} of folds {} is Complete'
              .format(self.DataSet,
                      self.target_kind,
                      str(self.folds)))

        print('{} Slides, with X{} magnification. {} tiles per iteration, {} iterations to complete full inference'
              .format(len(self.image_file_names),
                      self.desired_magnification,
                      self.tiles_per_iter,
                      self.__len__()))

    def __len__(self):
        return int(np.ceil(np.array(self.num_tiles) / self.tiles_per_iter).sum())

    def __getitem__(self, idx, location: List = None, tile_size: int = None):
        start_getitem = time.time()
        if idx == 0 or (idx > 0 and self.is_last_batch[idx - 1]):
            self.slide_num += 1

            if sys.platform == 'win32':
                image_file = os.path.join(self.image_path_names[self.slide_num],
                                          self.image_file_names[self.slide_num])
                self.current_slide = openslide.open_slide(image_file)
            else:
                self.current_slide = self.slides[self.slide_num]

            self.initial_num_patches = self.num_tiles[self.slide_num]
            if tile_size is not None:
                self.tile_size = tile_size

            '''desired_downsample = self.magnification[self.slide_num] / self.desired_magnification

            level, best_next_level = -1, -1
            for index, downsample in enumerate(self.current_slide.level_downsamples):
                if isclose(desired_downsample, downsample, rel_tol=1e-3):
                    level = index
                    level_downsample = 1
                    break

                elif downsample < desired_downsample:
                    best_next_level = index
                    level_downsample = int(
                        desired_downsample / self.current_slide.level_downsamples[best_next_level])

            self.adjusted_tile_size = self.tile_size * level_downsample
            self.best_slide_level = level if level > best_next_level else best_next_level
            self.level_0_tile_size = int(desired_downsample) * self.tile_size'''

            self.best_slide_level, self.adjusted_tile_size, self.level_0_tile_size = \
                get_optimal_slide_level(self.current_slide, self.magnification[self.slide_num],
                                        self.desired_magnification, self.tile_size)

        label = get_label(self.target[self.slide_num], self.multi_target)
        label = torch.LongTensor(label)

        tile_locations = self.slide_grids[idx] if location is None else location

        tiles, time_list, _ = _get_tiles(slide=self.current_slide,
                                         locations=tile_locations,
                                         tile_size_level_0=self.level_0_tile_size,
                                         adjusted_tile_sz=self.adjusted_tile_size,
                                         output_tile_sz=self.tile_size,
                                         best_slide_level=self.best_slide_level)

        X = torch.zeros([len(tiles), 3, self.tile_size, self.tile_size])

        start_aug = time.time()
        for i in range(len(tiles)):
            X[i] = self.transform(tiles[i])

        aug_time = time.time() - start_aug
        total_time = time.time() - start_getitem
        if self.print_time:
            time_list = (time_list[0], time_list[1], aug_time, total_time)
        else:
            time_list = [0]

        return {'Data': X,
                'Label': label,
                'Time List': time_list,
                'Is Last Batch': self.is_last_batch[idx],
                'Initial Num Tiles': self.initial_num_patches,
                'Slide Filename': self.current_slide._filename,
                'Equivalent Grid': self.equivalent_grid[idx],
                'Is Tissue Tiles': self.is_tissue_tiles[idx],
                'Equivalent Grid Size': self.equivalent_grid_size[self.slide_num],
                'Level 0 Locations': self.slide_grids[idx],
                'Original Data': transforms.ToTensor()(tiles[0])
                }


class Infer_Dataset_Background(WSI_Master_Dataset):
    """
    This dataset was created on 21/7/2021 to support extraction of background tiles in order to check the features and
    scores that they get from the model
    """

    def __init__(self,
                 DataSet: str = 'TCGA',
                 tile_size: int = 256,
                 tiles_per_iter: int = 500,
                 target_kind: str = 'ER',
                 folds: List = [1],
                 num_tiles: int = 500,
                 dx: bool = False,
                 desired_slide_magnification: int = 10
                 ):
        super(Infer_Dataset_Background, self).__init__(DataSet=DataSet,
                                                       tile_size=tile_size,
                                                       bag_size=None,
                                                       target_kind=target_kind,
                                                       test_fold=1,
                                                       infer_folds=folds,
                                                       train=True,
                                                       print_timing=False,
                                                       transform_type='none',
                                                       DX=dx,
                                                       get_images=False,
                                                       train_type='Infer',
                                                       desired_slide_magnification=desired_slide_magnification)

        self.tiles_per_iter = tiles_per_iter
        self.folds = folds
        self.magnification = []
        self.num_tiles = []
        self.slide_grids = []
        self.grid_lists = []
        self.patient_barcode = []

        ind = 0
        slide_with_not_enough_tiles = 0
        for _, slide_num in enumerate(self.valid_slide_indices):
            if (self.DX and self.all_is_DX_cut[slide_num]) or not self.DX:
                if num_tiles <= self.all_tissue_tiles[slide_num] and self.all_tissue_tiles[slide_num] > 0:
                    self.num_tiles.append(num_tiles)
                else:
                    self.num_tiles.append(int(self.all_tissue_tiles[slide_num]))
                    slide_with_not_enough_tiles += 1

                self.magnification.extend([self.all_magnifications[slide_num]])
                self.patient_barcode.append(self.all_patient_barcodes[slide_num])
                which_patches = sample(range(int(self.tissue_tiles[ind])), self.num_tiles[-1])

                if self.presaved_tiles[ind]:
                    self.grid_lists.append(0)
                else:
                    basic_file_name = '.'.join(self.image_file_names[ind].split('.')[:-1])
                    grid_file = os.path.join(self.image_path_names[ind], 'Grids_' + str(self.desired_magnification),
                                             basic_file_name + '--tlsz' + str(self.tile_size) + '.data')
                    with open(grid_file, 'rb') as filehandle:
                        grid_list = pickle.load(filehandle)
                        self.grid_lists.append(grid_list)

                patch_ind_chunks = chunks(which_patches, self.tiles_per_iter)
                self.slide_grids.extend(patch_ind_chunks)

                ind += 1

        print('There are {} slides with less than {} tiles'.format(slide_with_not_enough_tiles, num_tiles))

        # The following properties will be used in the __getitem__ function
        self.tiles_to_go = None
        self.slide_num = -1
        self.current_file = None

        print('Initiation of WSI INFERENCE for {} DataSet and {} of folds {} is Complete'
              .format(self.DataSet,
                      self.target_kind,
                      str(self.folds)))

        print('{} Slides, with X{} magnification. {} tiles per iteration, {} iterations to complete full inference'
              .format(len(self.image_file_names),
                      self.desired_magnification,
                      self.tiles_per_iter,
                      self.__len__()))

    def __len__(self):
        return int(np.ceil(np.array(self.num_tiles) / self.tiles_per_iter).sum())

    def __getitem__(self, idx):
        start_getitem = time.time()
        if self.tiles_to_go is None:
            self.slide_num += 1
            self.tiles_to_go = self.num_tiles[self.slide_num]

            self.current_slide = self.slides[self.slide_num]

            self.initial_num_patches = self.num_tiles[self.slide_num]

            if not self.presaved_tiles[self.slide_num]:
                self.best_slide_level, self.adjusted_tile_size, self.level_0_tile_size = \
                    get_optimal_slide_level(self.current_slide, self.magnification[self.slide_num],
                                            self.desired_magnification, self.tile_size)

        label = get_label(self.target[self.slide_num], self.multi_target)
        label = torch.LongTensor(label)

        if self.presaved_tiles[self.slide_num]:
            idxs = self.slide_grids[idx]
            empty_image = Image.fromarray(np.uint8(np.zeros((self.tile_size, self.tile_size, 3))))
            tiles = [empty_image] * len(idxs)
            for ii, tile_ind in enumerate(idxs):
                tile_path = os.path.join(self.slides[self.slide_num], 'tile_' + str(tile_ind) + '.data')
                with open(tile_path, 'rb') as fh:
                    header = fh.readline()
                    tile_bin = fh.read()
                dtype, w, h, c = header.decode('ascii').strip().split()
                tile = np.frombuffer(tile_bin, dtype=dtype).reshape((int(w), int(h), int(c)))
                tile1 = self.rand_crop(Image.fromarray(tile))
                tiles[ii] = tile1
        else:
            locs = [self.grid_lists[self.slide_num][loc] for loc in self.slide_grids[idx]]
            tiles, time_list, _ = _get_tiles(slide=self.current_slide,
                                             # locations=self.slide_grids[idx],
                                             locations=locs,
                                             tile_size_level_0=self.level_0_tile_size,
                                             adjusted_tile_sz=self.adjusted_tile_size,
                                             output_tile_sz=self.tile_size,
                                             best_slide_level=self.best_slide_level,
                                             random_shift=False)

        if self.tiles_to_go <= self.tiles_per_iter:
            self.tiles_to_go = None
        else:
            self.tiles_to_go -= self.tiles_per_iter

        X = torch.zeros([len(tiles), 3, self.tile_size, self.tile_size])

        start_aug = time.time()
        for i in range(len(tiles)):
            X[i] = self.transform(tiles[i])

        aug_time = time.time() - start_aug
        total_time = time.time() - start_getitem
        if self.print_time:
            time_list = (time_list[0], time_list[1], aug_time, total_time)
        else:
            time_list = [0]
        if self.tiles_to_go is None:
            last_batch = True
        else:
            last_batch = False

        return X, label, time_list, last_batch, self.initial_num_patches, self.image_file_names[self.slide_num], \
               self.patient_barcode[self.slide_num]


class WSI_Segmentation_Master_Dataset(Dataset):
    def __init__(self,
                 DataSet: str = 'TCGA',
                 tile_size: int = 256,
                 test_fold: int = 1,
                 infer_folds: List = [None],
                 train: bool = True,
                 print_timing: bool = False,
                 transform_type: str = 'flip',
                 get_images: bool = False,
                 train_type: str = 'MASTER',
                 color_param: float = 0.1,
                 n_tiles: int = 10,
                 desired_slide_magnification: int = 10,
                 slide_repetitions: int = 1):

        print('Initializing {} DataSet for Segmentation....'.format('Train' if train else 'Test'))

        self.DataSet = DataSet
        self.desired_magnification = desired_slide_magnification
        self.tile_size = tile_size
        self.test_fold = test_fold
        self.train = train
        self.print_time = print_timing
        self.get_images = get_images
        self.train_type = train_type
        self.color_param = color_param

        # Get DataSets location:
        self.dir_dict = get_datasets_dir_dict(Dataset=self.DataSet)
        print('Slide Data will be taken from these locations:')
        print(self.dir_dict)
        locations_list = []

        for _, key in enumerate(self.dir_dict):
            locations_list.append(self.dir_dict[key])

            slide_meta_data_file = os.path.join(self.dir_dict[key], 'slides_data_' + key + '.xlsx')
            grid_meta_data_file = os.path.join(self.dir_dict[key], 'Grids_' + str(self.desired_magnification),
                                               'Grid_data.xlsx')

            slide_meta_data_DF = pd.read_excel(slide_meta_data_file)
            grid_meta_data_DF = pd.read_excel(grid_meta_data_file)
            meta_data_DF = pd.DataFrame({**slide_meta_data_DF.set_index('file').to_dict(),
                                         **grid_meta_data_DF.set_index('file').to_dict()})

            self.meta_data_DF = meta_data_DF if not hasattr(self, 'meta_data_DF') else self.meta_data_DF.append(
                meta_data_DF)
        self.meta_data_DF.reset_index(inplace=True)
        self.meta_data_DF.rename(columns={'index': 'file'}, inplace=True)

        if self.DataSet == 'LUNG':
            # for lung, take only origin: lung
            self.meta_data_DF = self.meta_data_DF[self.meta_data_DF['Origin'] == 'lung']
            self.meta_data_DF.reset_index(inplace=True)

        all_targets = list(self.meta_data_DF[self.target_kind + ' status'])
        all_patient_barcodes = list(self.meta_data_DF['patient barcode'])

        # We'll use only the valid slides - the ones with a Negative or Positive label. (Some labels have other values)
        # Let's compute which slides are these:
        valid_slide_indices = np.where(np.isin(np.array(all_targets), ['Positive', 'Negative']) == True)[0]

        # Also remove slides without grid data:
        slides_without_grid = set(self.meta_data_DF.index[self.meta_data_DF['Total tiles - ' + str(
            self.tile_size) + ' compatible @ X' + str(self.desired_magnification)] == -1])
        # Remove slides with 0 tiles:
        slides_with_0_tiles = set(self.meta_data_DF.index[self.meta_data_DF['Legitimate tiles - ' + str(
            self.tile_size) + ' compatible @ X' + str(self.desired_magnification)] == 0])

        if 'bad segmentation' in self.meta_data_DF.columns:
            slides_with_bad_seg = set(self.meta_data_DF.index[self.meta_data_DF['bad segmentation'] == 1])
        else:
            slides_with_bad_seg = set()

        # Define number of tiles to be used
        if train_type == 'REG':
            n_minimal_tiles = n_tiles
        else:
            n_minimal_tiles = self.bag_size

        slides_with_few_tiles = set(self.meta_data_DF.index[self.meta_data_DF['Legitimate tiles - ' + str(
            self.tile_size) + ' compatible @ X' + str(self.desired_magnification)] < n_minimal_tiles])
        # FIXME: find a way to use slides with less than the minimal amount of slides. and than delete the following if.
        if len(slides_with_few_tiles) > 0:
            print(
                '{} Slides were excluded from DataSet because they had less than {} available tiles or are non legitimate for training'
                    .format(len(slides_with_few_tiles), n_minimal_tiles))
        valid_slide_indices = np.array(
            list(set(
                valid_slide_indices) - slides_without_grid - slides_with_few_tiles - slides_with_0_tiles - slides_with_bad_seg))

        # The train set should be a combination of all sets except the test set and validation set:
        if self.DataSet == 'CAT' or self.DataSet == 'ABCTB_TCGA':
            fold_column_name = 'test fold idx breast'
        else:
            fold_column_name = 'test fold idx'

        if self.train_type in ['REG', 'MIL']:
            if self.train:
                folds = list(self.meta_data_DF[fold_column_name].unique())
                folds.remove(self.test_fold)
                if 'test' in folds:
                    folds.remove('test')
                if 'val' in folds:
                    folds.remove('val')
            else:
                folds = [self.test_fold]
                folds.append('val')
        elif self.train_type == 'Infer':
            folds = infer_folds
        else:
            raise ValueError('Variable train_type is not defined')

        self.folds = folds

        correct_folds = self.meta_data_DF[fold_column_name][valid_slide_indices].isin(folds)
        valid_slide_indices = np.array(correct_folds.index[correct_folds])

        all_image_file_names = list(self.meta_data_DF['file'])

        all_image_ids = list(self.meta_data_DF['id'])

        all_in_fold = list(self.meta_data_DF[fold_column_name])
        all_tissue_tiles = list(self.meta_data_DF['Legitimate tiles - ' + str(self.tile_size) + ' compatible @ X' + str(
            self.desired_magnification)])

        if 'TCGA' not in self.dir_dict:
            self.DX = False
        if self.DX:
            all_is_DX_cut = list(self.meta_data_DF['DX'])

        all_magnifications = list(self.meta_data_DF['Manipulated Objective Power'])

        if train_type == 'Infer':
            self.valid_slide_indices = valid_slide_indices
            self.all_magnifications = all_magnifications
            self.all_is_DX_cut = all_is_DX_cut if self.DX else [True] * len(self.all_magnifications)
            self.all_tissue_tiles = all_tissue_tiles
            self.all_image_file_names = all_image_file_names
            self.all_patient_barcodes = all_patient_barcodes

        self.image_file_names = []
        self.image_path_names = []
        self.in_fold = []
        self.tissue_tiles = []
        self.target = []
        self.magnification = []
        self.slides = []
        self.grid_lists = []
        self.presaved_tiles = []

        for _, index in enumerate(tqdm(valid_slide_indices)):
            if (self.DX and all_is_DX_cut[index]) or not self.DX:
                self.image_file_names.append(all_image_file_names[index])
                # self.image_path_names.append(all_image_path_names[index])
                self.image_path_names.append(self.dir_dict[all_image_ids[index]])
                self.in_fold.append(all_in_fold[index])
                self.tissue_tiles.append(all_tissue_tiles[index])
                self.target.append(all_targets[index])
                self.magnification.append(all_magnifications[index])
                # self.presaved_tiles.append(all_image_ids[index] == 'ABCTB')
                self.presaved_tiles.append(all_image_ids[index] == 'ABCTB_TILES')

                # Preload slides - improves speed during training.
                try:
                    image_file = os.path.join(self.dir_dict[all_image_ids[index]], all_image_file_names[index])
                    if self.presaved_tiles[-1]:
                        tiles_dir = os.path.join(self.dir_dict[all_image_ids[index]], 'tiles',
                                                 '.'.join((os.path.basename(image_file)).split('.')[:-1]))
                        self.slides.append(tiles_dir)
                        self.grid_lists.append(0)
                    else:
                        self.slides.append(openslide.open_slide(image_file))
                        basic_file_name = '.'.join(all_image_file_names[index].split('.')[:-1])
                        grid_file = os.path.join(self.dir_dict[all_image_ids[index]],
                                                 'Grids_' + str(self.desired_magnification),
                                                 basic_file_name + '--tlsz' + str(self.tile_size) + '.data')
                        with open(grid_file, 'rb') as filehandle:
                            grid_list = pickle.load(filehandle)
                            self.grid_lists.append(grid_list)
                except FileNotFoundError:
                    raise FileNotFoundError(
                        'Couldn\'t open slide {} or it\'s Grid file {}'.format(image_file, grid_file))

        # Setting the transformation:
        self.transform = define_transformations(transform_type, self.train, self.tile_size, self.color_param)
        if np.sum(self.presaved_tiles):
            self.rand_crop = transforms.RandomCrop(self.tile_size)

        if train_type == 'REG':
            self.factor = n_tiles
            self.real_length = int(self.__len__() / self.factor)
        elif train_type == 'MIL':
            self.factor = 1
            self.real_length = self.__len__()
        if train is False and test_time_augmentation:
            self.factor = 4
            self.real_length = int(self.__len__() / self.factor)

    def __len__(self):
        return len(self.target) * self.factor

    def __getitem__(self, idx):
        start_getitem = time.time()
        idx = idx % self.real_length

        if self.presaved_tiles[idx]:  # load presaved patches
            time_tile_extraction = time.time()
            idxs = sample(range(self.tissue_tiles[idx]), self.bag_size)
            empty_image = Image.fromarray(np.uint8(np.zeros((self.tile_size, self.tile_size, 3))))
            tiles = [empty_image] * self.bag_size
            for ii, tile_ind in enumerate(idxs):
                tile_path = os.path.join(self.slides[idx], 'tile_' + str(tile_ind) + '.data')
                with open(tile_path, 'rb') as fh:
                    header = fh.readline()
                    tile_bin = fh.read()
                dtype, w, h, c = header.decode('ascii').strip().split()
                tile = np.frombuffer(tile_bin, dtype=dtype).reshape((int(w), int(h), int(c)))
                tile1 = self.rand_crop(Image.fromarray(tile))
                tiles[ii] = tile1
                time_tile_extraction = (time.time() - time_tile_extraction) / len(idxs)
                time_list = [0, time_tile_extraction]
        else:
            slide = self.slides[idx]

            tiles, time_list, _, _ = _choose_data(grid_list=self.grid_lists[idx],
                                                  slide=slide,
                                                  how_many=self.bag_size,
                                                  magnification=self.magnification[idx],
                                                  tile_size=self.tile_size,
                                                  print_timing=self.print_time,
                                                  desired_mag=self.desired_magnification,
                                                  random_shift=self.random_shift)

        label = get_label(self.target[idx])
        label = torch.LongTensor(label)

        # X will hold the images after all the transformations
        X = torch.zeros([self.bag_size, 3, self.tile_size, self.tile_size])

        start_aug = time.time()
        for i in range(self.bag_size):
            X[i] = self.transform(tiles[i])

        if self.get_images:
            images = torch.zeros([self.bag_size, 3, self.tile_size, self.tile_size])
            trans = transforms.Compose([transforms.CenterCrop(self.tile_size), transforms.ToTensor()])
            for i in range(self.bag_size):
                images[i] = trans(tiles[i])
        else:
            images = 0

        aug_time = (time.time() - start_aug) / self.bag_size
        total_time = time.time() - start_getitem
        if self.print_time:
            time_list = (time_list[0], time_list[1], aug_time, total_time)
        else:
            time_list = [0]

        return X, label, time_list, self.image_file_names[idx], images


class Features_to_Survival_Old(Dataset):
    def __init__(self,
                 is_train: bool = True,
                 is_per_patient: bool = False,
                 is_per_slide: bool = False,
                 is_all_tiles: bool = False,
                 bag_size: int = 1,
                 is_all_censored: bool = False,
                 is_all_not_censored: bool = False,
                 ):

        if is_all_censored and is_all_not_censored:
            raise Exception('\'is_all_censored\' and \'is_all_not_censored\' CANNOT be TRUE at the same time')

        if is_per_patient and is_per_slide:
            raise Exception('Data arrangement cannot be "per slide" and "per patient" at the same time')

        if sys.platform == 'darwin':
            if is_train:
                data_location = r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/survival/Ran_Exp_20094-survival-TestFold_1/Train/'
            else:
                data_location = r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/survival/Ran_Exp_20094-survival-TestFold_1/Test/'

        elif sys.platform == 'linux':
            if is_train:
                data_location = r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20094-survival-TestFold_1/Inference/train_w_features_new/'
            else:
                data_location = r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20094-survival-TestFold_1/Inference/test_w_features_new/'

        dataset = 'ABCTB'
        self.train_type = 'Features'
        self.is_train = is_train

        self.slide_names = []
        self.labels = []
        # self.targets = []
        self.features = []
        self.num_tiles = []
        self.scores = []
        self.tile_scores = []
        self.tile_location = []
        self.patient_data = {}
        self.bad_patient_list = []
        self.is_per_patient, self.is_per_slide = is_per_patient, is_per_slide
        self.is_all_tiles = is_all_tiles
        self.bag_size = bag_size
        self.censor, self.time_target, self.binary_target = [], [], []

        slides_from_same_patient_with_different_target_values, total_slides, bad_num_of_good_tiles = 0, 0, 0
        slides_with_not_enough_tiles, slides_with_bad_segmentation = 0, 0
        patient_list = []

        data_files = glob(os.path.join(data_location, '*.data'))

        print('Loading data from files in location: {}'.format(data_location))
        corrected_data_file_list = []
        for data_file in data_files:
            if 'features' in data_file.split('/')[-1].split('_'):
                corrected_data_file_list.append(data_file)

        data_files = corrected_data_file_list

        if sys.platform == 'darwin':
            if dataset == 'ABCTB':
                grid_location_dict = {
                    'ABCTB': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/Grids_data/ABCTB_TIF_Grid_data.xlsx'}
                slide_data_DF_dict = {'ABCTB': pd.read_excel(
                    '/Users/wasserman/Developer/WSI_MIL/All Data/Features/Grids_data/slides_data_ABCTB.xlsx')}

            else:
                raise Exception("Need to write which dictionaries to use in this receptor case")

        elif sys.platform == 'linux':
            if dataset == 'ABCTB':
                grid_location_dict = {'ABCTB': r'/mnt/gipmed_new/Data/ABCTB_TIF/Grids_10/Grid_data.xlsx'}
                slide_data_DF_dict = {'ABCTB': pd.read_excel(r'/mnt/gipmed_new/Data/ABCTB_TIF/slides_data_ABCTB.xlsx')}

            else:
                raise Exception("Need to write which dictionaries to use in this receptor case")

        grid_DF = pd.DataFrame()
        slide_data_DF = pd.DataFrame()
        for key in grid_location_dict.keys():
            new_grid_DF = pd.read_excel(grid_location_dict[key])
            grid_DF = pd.concat([grid_DF, new_grid_DF])
            slide_data_DF = pd.concat([slide_data_DF, slide_data_DF_dict[key]])

        grid_DF.set_index('file', inplace=True)
        slide_data_DF.set_index('file', inplace=True)

        for file_idx, file in enumerate(tqdm(data_files)):
            with open(file, 'rb') as filehandle:
                inference_data = pickle.load(filehandle)

            try:
                if len(inference_data) == 6:
                    labels, targets, scores, patch_scores, slide_names, features = inference_data
                    tile_location = np.array([[(np.nan, np.nan)] * patch_scores.shape[1]] * patch_scores.shape[0])
                elif len(inference_data) == 7:
                    labels, targets, scores, patch_scores, slide_names, features, batch_number = inference_data
                    tile_location = np.array([[np.nan, np.nan] * patch_scores.shape[1]] * patch_scores.shape[0])
                elif len(inference_data) == 8:
                    labels, targets, scores, patch_scores, slide_names, features, batch_number, tile_location = inference_data
            except ValueError:
                raise Exception('Debug')

            try:
                num_slides, max_tile_num = features.shape[0], features.shape[2]
            except UnboundLocalError:
                print(file_idx, file, len(inference_data))
                print(features.shape)

            for slide_num in range(num_slides):
                # Skip slides that have a "bad segmentation" marker in GridData.xlsx file
                try:
                    if grid_DF.loc[slide_names[slide_num], 'bad segmentation'] == 1:
                        slides_with_bad_segmentation += 1
                        continue
                except ValueError:
                    raise Exception('Debug')
                except KeyError:
                    raise Exception('Debug')

                # Skip slides that have a "Exclude" marker in Slides_data.xlsx file
                if slide_data_DF.loc[slide_names[slide_num]]['Exclude for time prediction?'] == 'Exclude':
                    continue

                total_slides += 1
                feature_1 = features[slide_num, :, :, 0]
                nan_indices = np.argwhere(np.isnan(feature_1)).tolist()
                tiles_in_slide = nan_indices[0][1] if bool(
                    nan_indices) else max_tile_num  # check if there are any nan values in feature_1
                column_title = 'Legitimate tiles - 256 compatible @ X10' if len(
                    dataset.split('_')) == 1 else 'Legitimate tiles - 256 compatible @ X' + dataset.split('_')[1]
                try:
                    tiles_in_slide_from_grid_data = int(grid_DF.loc[slide_names[slide_num], column_title])
                except TypeError:
                    raise Exception('Debug')

                # Get censor status and targets (Binary and time).
                censor_status = slide_data_DF.loc[slide_names[slide_num]]['Censored']
                censor_status = num_2_bool(censor_status)
                target_time = slide_data_DF.loc[slide_names[slide_num]]['Follow-up Months Since Diagnosis']
                target_binary = slide_data_DF.loc[slide_names[slide_num]]['survival status']
                target_binary = get_label(target_binary)[0]

                # Check that binary target from feature files matched the data in slides_data.xlsx:
                if target_binary != targets[slide_num]:
                    raise Exception('Mismatch found between targets')

                if is_per_patient:
                    # calculate patient id:
                    patient = slide_names[slide_num].split('.')[0]
                    '''if patient.split('-')[0] == 'TCGA':
                        patient = '-'.join(patient.split('-')[:3])
                    elif slide_names[slide_num].split('.')[-1] == 'mrxs':  # This is a CARMEL slide
                        #patient = slides_data_DF_CARMEL.loc[slide_names[slide_num], 'patient barcode']
                        patient = slide_data_DF.loc[slide_names[slide_num], 'patient barcode']'''

                    # insert to the "all patient list"
                    patient_list.append(patient)

                    # Check if the patient has already been observed to be with multiple targets.
                    # if so count the slide as bad slide and move on to the next slide
                    if patient in self.bad_patient_list:
                        slides_from_same_patient_with_different_target_values += 1
                        continue

                    # in case this patient has already been seen, than it has multiple slides
                    if patient in self.patient_data.keys():
                        patient_dict = self.patient_data[patient]

                        # Check if the patient has multiple targets
                        patient_same_target = True if int(targets[slide_num]) == patient_dict[
                            'target'] else False  # Checking the the patient target is not changing between slides
                        # if the patient has multiple targets than we need to remove it from the valid data:
                        if not patient_same_target:
                            slides_from_same_patient_with_different_target_values += 1 + len(patient_dict[
                                                                                                 'slides'])  # we skip more than 1 slide since we need to count the current slide and the ones that are already inserted to the patient_dict
                            self.patient_data.pop(patient)  # remove it from the dictionary of legitimate patients
                            self.bad_patient_list.append(patient)  # insert it to the list of non legitimate patients
                            continue  # and move on to the next slide

                        patient_dict['num tiles'].append(tiles_in_slide)
                        patient_dict['tile scores'] = np.concatenate(
                            (patient_dict['tile scores'], patch_scores[slide_num, :tiles_in_slide]), axis=0)
                        patient_dict['labels'].append(int(labels[slide_num]))
                        # A patient with multiple slides has only 1 target, therefore another target should not be inserted into the dict
                        # patient_dict['target'].append(int(targets[slide_num]))
                        patient_dict['slides'].append(slide_names[slide_num])
                        patient_dict['scores'].append(scores[slide_num])

                        features_old = patient_dict['features']
                        features_new = features[slide_num, :, :tiles_in_slide, :].squeeze(0).astype(np.float32)
                        patient_dict['features'] = np.concatenate((features_old, features_new), axis=0)

                        # Check that the censor status, target_time and target_binary has not changed:
                        if censor_status != self.patient_data[patient]['Censored']:
                            raise Exception('This patient has multiple censor status')
                        if target_binary != self.patient_data[patient]['Binary Target']:
                            raise Exception('This patient has multiple Binary Targets')
                        if target_time != self.patient_data[patient]['Time Target']:
                            raise Exception('This patient has multiple Time Targets')

                    else:
                        patient_dict = {'num tiles': [tiles_in_slide],
                                        'features': features[slide_num, :, :tiles_in_slide, :].squeeze(0).astype(
                                            np.float32),
                                        'tile scores': patch_scores[slide_num, :tiles_in_slide],
                                        'labels': [int(labels[slide_num])],
                                        'target': int(targets[slide_num]),
                                        'slides': [slide_names[slide_num]],
                                        'scores': [scores[slide_num]],
                                        'Censored': censor_status,
                                        'Time Target': target_time,
                                        'Binary Target': target_binary
                                        }

                        self.patient_data[patient] = patient_dict

                else:
                    if (is_all_not_censored and censor_status) or (
                            is_all_censored and not censor_status):  # FIXME: fix this line
                        continue

                    self.num_tiles.append(tiles_in_slide)
                    self.features.append(features[slide_num, :, :tiles_in_slide, :].squeeze(0).astype(np.float32))
                    self.tile_scores.append(patch_scores[slide_num, :tiles_in_slide])
                    self.slide_names.append(slide_names[slide_num])
                    self.labels.append(int(labels[slide_num]))
                    # self.targets.append(int(targets[slide_num]))
                    self.scores.append(scores[slide_num])
                    # self.tile_location.append(tile_location[slide_num, :tiles_in_slide, :])
                    self.tile_location.append(tile_location[slide_num, :tiles_in_slide])

                    self.censor.append(censor_status)
                    self.time_target.append(target_time)
                    self.binary_target.append(target_binary)

        print('There are {}/{} slides with \"bad number of good tile\" '.format(bad_num_of_good_tiles, total_slides))
        print('There are {}/{} slides with \"bad segmentation\" '.format(slides_with_bad_segmentation, total_slides))

        if self.is_per_patient:
            self.patient_keys = list(self.patient_data.keys())
            print('Skipped {}/{} slides for {}/{} patients (Inconsistent target value for same patient)'.format(
                slides_from_same_patient_with_different_target_values, total_slides, len(self.bad_patient_list),
                len(set(patient_list))))
            print('Initialized Dataset with {} feature slides in {} patients'.format(
                total_slides - slides_from_same_patient_with_different_target_values - slides_with_not_enough_tiles,
                self.__len__()))
        else:
            print('Initialized Dataset with {} feature slides'.format(len(self.censor)))

    def __len__(self):
        if self.is_per_patient:
            return len(self.patient_keys)

        elif self.is_per_slide:
            return len(self.slide_names)

        else:
            if self.is_all_tiles:
                return len(self.slide_names)
            else:
                return len(self.slide_names) * 10

    def __getitem__(self, item):
        if self.is_per_patient:
            patient_data = self.patient_data[self.patient_keys[item]]

            return {'Binary Target': patient_data['Binary Target'],
                    'Time Target': patient_data['Time Target'],
                    'Features': patient_data['features'],
                    'Censored': patient_data['Censored']
                    }

        else:
            if not self.is_all_tiles:  # In case we're not getting all the tiles of a slide we want to fo over each slide for 10 times during each epoch
                item = item % len(self.censor)
            if self.is_per_slide:
                tile_idx = list(range(self.num_tiles[item]))
            else:
                tile_idx = list(range(self.num_tiles[item])) if self.is_all_tiles else choices(
                    range(self.num_tiles[item]), k=self.bag_size)

            return {'Binary Target': self.binary_target[item],
                    'Time Target': self.time_target[item],
                    'Features': self.features[item][tile_idx],
                    'Censored': self.censor[item],
                    'tile scores': self.tile_scores[item][tile_idx],
                    'slide name': self.slide_names[item],
                    'num tiles': self.num_tiles[item],
                    'tile locations': self.tile_location[item][tile_idx] if hasattr(self, 'tile_location') else None
                    }


class Features_to_Survival(Dataset):
    def __init__(self,
                 is_train: bool = True,
                 # is_per_patient: bool = False,
                 # is_per_slide: bool = False,
                 is_all_tiles: bool = False,
                 bag_size: int = 1,
                 use_patient_features: bool = False,
                 normalized_features: bool = False,
                 is_all_censored: bool = False,
                 is_all_not_censored: bool = False,
                 ):

        if is_all_censored and is_all_not_censored:
            raise Exception('\'is_all_censored\' and \'is_all_not_censored\' CANNOT be TRUE at the same time')

        if sys.platform == 'darwin':
            if use_patient_features:
                if is_train:
                    data_location = r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/survival/Ran_Exp_20094-survival-TestFold_1/Augmented/Train/'
                else:
                    data_location = r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/survival/Ran_Exp_20094-survival-TestFold_1/Augmented/Test/'

            elif normalized_features:
                if is_train:
                    data_location = r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/survival/Ran_Exp_20094-survival-TestFold_1/Normalized/Train/'
                else:
                    data_location = r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/survival/Ran_Exp_20094-survival-TestFold_1/Normalized/Test/'

            else:
                if is_train:
                    data_location = r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/survival/Ran_Exp_20094-survival-TestFold_1/Train/'
                else:
                    data_location = r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/survival/Ran_Exp_20094-survival-TestFold_1/Test/'

        elif sys.platform == 'linux':
            if is_train:
                data_location = r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20094-survival-TestFold_1/Inference/train_w_features_new/'
            else:
                data_location = r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20094-survival-TestFold_1/Inference/test_w_features_new/'

        dataset = 'ABCTB'
        self.train_type = 'Features'
        self.is_train = is_train

        self.slide_names = []
        self.patient_id = []
        self.slide_for_patient_with_multiple_targets = []
        self.labels = []
        self.features = []
        self.num_tiles = []
        self.scores = []
        self.tile_scores = []
        self.tile_location = []
        self.bad_patient_list = []
        self.patient_data = {}
        #  self.is_per_patient, self.is_per_slide = is_per_patient, is_per_slide
        self.is_all_tiles = is_all_tiles
        self.bag_size = bag_size
        self.censor, self.time_target, self.binary_target = [], [], []

        slides_from_same_patient_with_different_target_values, total_slides, bad_num_of_good_tiles = 0, 0, 0
        slides_with_not_enough_tiles, slides_with_bad_segmentation = 0, 0
        patient_list = []

        data_files = glob(os.path.join(data_location, '*.data'))

        print('Loading data from files in location: {}'.format(data_location))
        corrected_data_file_list = []
        for data_file in data_files:
            if 'features' in data_file.split('/')[-1].split('_'):
                corrected_data_file_list.append(data_file)

        data_files = corrected_data_file_list

        if sys.platform == 'darwin':
            if dataset == 'ABCTB':
                grid_location_dict = {
                    'ABCTB': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/Grids_data/ABCTB_TIF_Grid_data.xlsx'}
                slide_data_DF_dict = {'ABCTB': pd.read_excel(
                    '/Users/wasserman/Developer/WSI_MIL/All Data/Features/Grids_data/slides_data_ABCTB.xlsx')}
                patient_feature_data_dict = {'ABCTB': pd.read_excel(
                    '/Users/wasserman/Developer/WSI_MIL/All Data/Features/Grids_data/ABCTB_Path_Data_features.xlsx')}
                data_stat_DF = pd.read_excel(
                    r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/Grids_data/Mean_and_STD_for_features.xlsx')
                mean_vector = data_stat_DF['Mean'].to_numpy()
                std_vector = data_stat_DF['STD'].to_numpy()

            else:
                raise Exception("Need to write which dictionaries to use in this receptor case")

        elif sys.platform == 'linux':
            if dataset == 'ABCTB':
                grid_location_dict = {'ABCTB': r'/mnt/gipmed_new/Data/ABCTB_TIF/Grids_10/Grid_data.xlsx'}
                slide_data_DF_dict = {'ABCTB': pd.read_excel(r'/mnt/gipmed_new/Data/ABCTB_TIF/slides_data_ABCTB.xlsx')}

            else:
                raise Exception("Need to write which dictionaries to use in this receptor case")

        grid_DF = pd.DataFrame()
        slide_data_DF = pd.DataFrame()
        patient_feature_data_DF = pd.DataFrame()

        for key in grid_location_dict.keys():
            new_grid_DF = pd.read_excel(grid_location_dict[key])
            grid_DF = pd.concat([grid_DF, new_grid_DF])
            slide_data_DF = pd.concat([slide_data_DF, slide_data_DF_dict[key]])
            patient_feature_data_DF = pd.concat([patient_feature_data_DF, patient_feature_data_dict[key]])

        grid_DF.set_index('file', inplace=True)
        slide_data_DF.set_index('file', inplace=True)
        patient_feature_data_DF.set_index('Identifier', inplace=True)

        for file_idx, file in enumerate(tqdm(data_files)):
            with open(file, 'rb') as filehandle:
                inference_data = pickle.load(filehandle)

            try:
                if len(inference_data) == 6:
                    labels, targets, scores, patch_scores, slide_names, features = inference_data
                    tile_location = np.array([[(np.nan, np.nan)] * patch_scores.shape[1]] * patch_scores.shape[0])
                elif len(inference_data) == 7:
                    labels, targets, scores, patch_scores, slide_names, features, batch_number = inference_data
                    tile_location = np.array([[np.nan, np.nan] * patch_scores.shape[1]] * patch_scores.shape[0])
                elif len(inference_data) == 8:
                    labels, targets, scores, patch_scores, slide_names, features, batch_number, tile_location = inference_data
            except ValueError:
                raise Exception('Debug')

            try:
                num_slides, max_tile_num = features.shape[0], features.shape[2]
            except UnboundLocalError:
                print(file_idx, file, len(inference_data))
                print(features.shape)

            for slide_num in range(num_slides):
                # Skip slides that have a "bad segmentation" marker in GridData.xlsx file
                try:
                    if grid_DF.loc[slide_names[slide_num], 'bad segmentation'] == 1:
                        slides_with_bad_segmentation += 1
                        continue
                except ValueError:
                    raise Exception('Debug')
                except KeyError:
                    raise Exception('Debug')

                # Skip slides that have a "Exclude" marker in Slides_data.xlsx file
                if slide_data_DF.loc[slide_names[slide_num]]['Exclude for time prediction?'] == 'Exclude':
                    continue

                total_slides += 1
                feature_1 = features[slide_num, :, :, 0]
                nan_indices = np.argwhere(np.isnan(feature_1)).tolist()
                tiles_in_slide = nan_indices[0][1] if bool(
                    nan_indices) else max_tile_num  # check if there are any nan values in feature_1
                column_title = 'Legitimate tiles - 256 compatible @ X10' if len(
                    dataset.split('_')) == 1 else 'Legitimate tiles - 256 compatible @ X' + dataset.split('_')[1]
                try:
                    tiles_in_slide_from_grid_data = int(grid_DF.loc[slide_names[slide_num], column_title])
                except TypeError:
                    raise Exception('Debug')

                # Get censor status and targets (Binary and time).
                censor_status = slide_data_DF.loc[slide_names[slide_num]]['Censored']
                censor_status = num_2_bool(censor_status)
                target_time = slide_data_DF.loc[slide_names[slide_num]]['Follow-up Months Since Diagnosis']
                target_binary = slide_data_DF.loc[slide_names[slide_num]]['survival status']
                target_binary = get_label(target_binary)[0]

                # Check that binary target from feature files matched the data in slides_data.xlsx:
                if target_binary != targets[slide_num]:
                    raise Exception('Mismatch found between targets')

                # calculate patient id:
                patient = slide_names[slide_num].split('.')[0]
                self.patient_id.append(patient)

                # Check if the patient has already been observed to be with multiple targets.
                # if so mark it as such and add to the count of those kind of slides.
                if patient in self.bad_patient_list:
                    self.slide_for_patient_with_multiple_targets.append(True)
                    slides_from_same_patient_with_different_target_values += 1
                else:
                    self.slide_for_patient_with_multiple_targets.append(False)

                # in case this patient has already been seen, than it has multiple slides and we need to check for target consistency:
                if patient in self.patient_data.keys():
                    self.patient_data[patient]['slides'].append(
                        slide_names[slide_num])  # Add the slide name to the slide of this patient

                    # Check if the patient has multiple targets
                    same_target_for_patient = True if int(targets[slide_num]) == self.patient_data[patient][
                        'Binary Target'] else False  # Checking the the patient target is not changing between slides
                    # if the patient has multiple targets than we need to mark it as so.
                    if not same_target_for_patient:
                        slides_from_same_patient_with_different_target_values += 1 + len(patient_dict[
                                                                                             'slides'])  # we skip more than 1 slide since we need to count the current slide and the ones that are already inserted to the patient_dict
                        self.bad_patient_list.append(patient)  # insert it to the list of non legitimate patients

                    # Check that the censor status and target_time has not changed:
                    if censor_status != self.patient_data[patient]['Censored']:
                        raise Exception('This patient has multiple censor status')
                    if target_time != self.patient_data[patient]['Time Target']:
                        raise Exception('This patient has multiple Time Targets')

                else:  # Its a new patient and we need to create a dict for it
                    patient_dict = {'slides': [slide_names[slide_num]],
                                    'Censored': censor_status,
                                    'Time Target': target_time,
                                    'Binary Target': target_binary
                                    }

                    self.patient_data[patient] = patient_dict

                if (is_all_not_censored and censor_status) or (
                        is_all_censored and not censor_status):  # FIXME: fix this line
                    continue

                self.num_tiles.append(tiles_in_slide)
                self.features.append(features[slide_num, :, :tiles_in_slide, :].squeeze(0).astype(np.float32))
                self.tile_scores.append(patch_scores[slide_num, :tiles_in_slide])
                self.slide_names.append(slide_names[slide_num])
                self.labels.append(int(labels[slide_num]))
                self.scores.append(scores[slide_num])
                self.tile_location.append(tile_location[slide_num, :tiles_in_slide])
                self.censor.append(censor_status)
                self.time_target.append(target_time)
                self.binary_target.append(target_binary)

        print('There are {}/{} slides with \"bad number of good tile\" '.format(bad_num_of_good_tiles, total_slides))
        print('There are {}/{} slides with \"bad segmentation\" '.format(slides_with_bad_segmentation, total_slides))

        print('There are {}/{} slides for {}/{} patients with Inconsistent target value for same patient)'.format(
            slides_from_same_patient_with_different_target_values,
            total_slides,
            len(self.bad_patient_list),
            len(self.patient_data.keys())))

        print('Initialized Dataset with {} feature slides'.format(len(self.censor)))

    def __len__(self):
        # The size of the train set is X 10 the number of slides so that each epoch will be X 10 the size of the slides
        # The size of the TestSet or when we want to get all tiles from each slide will be the number of the slides
        if self.is_all_tiles or not self.is_train:
            return len(self.slide_names)
        else:
            return len(self.slide_names) * 10

    def __getitem__(self, item):
        start_time = time.time()
        if not self.is_all_tiles:  # In case we're not getting all the tiles of a slide we want to fo over each slide for 10 times during each epoch
            item = item % len(self.censor)

        # For TestSet we'll get all the tiles for each slide since the length of the dataset is defined as the number of slides
        # For the trainset we need to rescale 'item'
        if not self.is_train:
            tile_idx = list(range(self.num_tiles[item]))
        else:
            tile_idx = list(range(self.num_tiles[item])) if self.is_all_tiles else choices(range(self.num_tiles[item]),
                                                                                           k=self.bag_size)

        to_return = {'Binary Target': self.binary_target[item],
                     'Time Target': self.time_target[item],
                     'Features': self.features[item][tile_idx],
                     'Censored': self.censor[item],
                     'tile scores': self.tile_scores[item][tile_idx],
                     'slide name': self.slide_names[item],
                     'num tiles': self.num_tiles[item],
                     'tile locations': self.tile_location[item][tile_idx] if hasattr(self, 'tile_location') else None,
                     'Slide belongs to Patient with Multiple Targets': self.slide_for_patient_with_multiple_targets[
                         item],
                     'Slide Belongs to Patient': self.patient_id[item],
                     'Time': -1
                     }

        to_return['Time'] = time.time() - start_time
        return to_return


class C_Index_Test_Dataset(Dataset):
    def __init__(self,
                 train: bool = True,
                 is_all_censored: bool = False,
                 is_all_not_censored: bool = False,
                 data_difficulty: 'str' = 'Basic'
                 ):

        if is_all_censored and is_all_not_censored:
            raise Exception('\'is_all_censored\' and \'is_all_not_censored\' CANNOT be TRUE at the same time')

        self.train_type = 'Features'

        if sys.platform == 'darwin':
            if data_difficulty == 'Basic':
                file_location = r'/Users/wasserman/Developer/WSI_MIL/All Data/survival_synthetic/Survival_time_synthetic.xlsx'
            elif data_difficulty == 'Moderate':
                file_location = r'/Users/wasserman/Developer/WSI_MIL/All Data/survival_synthetic/Survival_time_synthetic_80_censored.xlsx'
            elif data_difficulty == 'Difficult':
                file_location = r'/Users/wasserman/Developer/WSI_MIL/All Data/survival_synthetic/Survival_time_synthetic_hard.xlsx'
            else:
                raise Exception('Data difficulty is not identified')

        elif sys.platform == 'linux':
            file_location = r'/home/womer/project/All Data/survival_synthetic/Survival_time_synthetic.xlsx'
        DF = pd.read_excel(file_location)

        print("Loading data from file {}".format(file_location))
        num_cases = DF.shape[0]

        legit_cases = list(range(0, int(0.8 * num_cases)) if train else range(int(0.8 * num_cases), num_cases))

        self.data = []
        for case_index in tqdm(legit_cases):
            if (is_all_not_censored and DF.loc[case_index]['Censored']) or (
                    is_all_censored and not DF.loc[case_index]['Censored']):
                continue
            case = {'Binary Target': DF.loc[case_index]['Binary label'],
                    'Time Target': DF.loc[case_index, 'Observed survival time'],
                    'Features': np.array(DF.loc[case_index, [0, 1, 2, 3, 4, 5, 6, 7]]).astype(np.float32),
                    'Censored': DF.loc[case_index, 'Censored'],
                    'True Time Targets': DF.loc[case_index, 'True survival time']
                    }
            self.data.append(case)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


class C_Index_Test_Dataset_Original(Dataset):
    def __init__(self,
                 train: bool = True,
                 binary_target: bool = False,
                 without_censored: bool = False):

        self.train_type = 'Features'

        if sys.platform == 'darwin':
            # file_location = r'/Users/wasserman/Developer/WSI_MIL/All Data/survival_synthetic/Survival_time_synthetic.xlsx'
            file_location = r'/Users/wasserman/Developer/WSI_MIL/All Data/survival_synthetic/Survival_time_synthetic_hard.xlsx'

        elif sys.platform == 'linux':
            file_location = r'/home/womer/project/All Data/survival_synthetic/Survival_time_synthetic.xlsx'
        DF = pd.read_excel(file_location)

        num_cases = DF.shape[0]

        legit_cases = list(range(0, int(0.8 * num_cases)) if train else range(int(0.8 * num_cases), num_cases))

        all_targets = DF['Observed survival time']

        self.data = []
        for case_index in tqdm(legit_cases):
            if binary_target and (DF.loc[case_index]['Binary label'] == -1 or DF.loc[case_index, 'Censored']):
                # if DF.loc[case_index]['Binary label'] == -1 or DF.loc[case_index, 'Censored']:
                continue
            if without_censored and DF.loc[case_index]['Censored']:
                continue
            case = {'Binary Target': DF.loc[case_index]['Binary label'],
                    'Time Target': DF.loc[case_index, 'Observed survival time'],
                    'Features': np.array(DF.loc[case_index, [0, 1, 2, 3, 4, 5, 6, 7]]).astype(np.float32),
                    'Censored': DF.loc[case_index, 'Censored']
                    }
            self.data.append(case)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


class WSI_REGdataset_Survival(WSI_Master_Dataset_Survival):
    def __init__(self,
                 DataSet: str = 'TCGA',
                 tile_size: int = 256,
                 target_kind: str = 'ER',
                 test_fold: int = 1,
                 train: bool = True,
                 transform_type: str = 'flip',
                 DX: bool = False,
                 get_images: bool = False,
                 color_param: float = 0.1,
                 n_tiles: int = 10,
                 desired_slide_magnification: int = 10,
                 loan: bool = False,
                 er_eq_pr: bool = False,
                 slide_per_block: bool = False,
                 balanced_dataset: bool = False,
                 is_all_censored: bool = False,
                 is_all_not_censored: bool = False
                 ):
        super(WSI_REGdataset_Survival, self).__init__(DataSet=DataSet,
                                                      tile_size=tile_size,
                                                      bag_size=1,
                                                      target_kind=target_kind,
                                                      test_fold=test_fold,
                                                      train=train,
                                                      transform_type=transform_type,
                                                      DX=DX,
                                                      get_images=get_images,
                                                      train_type='REG',
                                                      color_param=color_param,
                                                      n_tiles=n_tiles,
                                                      desired_slide_magnification=desired_slide_magnification,
                                                      er_eq_pr=er_eq_pr,
                                                      slide_per_block=slide_per_block,
                                                      balanced_dataset=balanced_dataset,
                                                      is_all_censored=is_all_censored,
                                                      is_all_not_censored=is_all_not_censored
                                                      )

        self.loan = loan
        logging.info(
            'Initiation of WSI({}) {} {} DataSet for {} is Complete. Magnification is X{}, {} Slides, Tiles of size {}^2. {} tiles in a bag, {} Transform. TestSet is fold #{}. DX is {}'
                .format(self.train_type,
                        'Train' if self.train else 'Test',
                        self.DataSet,
                        self.target_kind,
                        self.desired_magnification,
                        self.real_length,
                        self.tile_size,
                        self.bag_size,
                        'Without' if transform_type == 'none' else 'With',
                        self.test_fold,
                        'ON' if self.DX else 'OFF'))

    def __getitem__(self, idx):
        data_dict = super(WSI_REGdataset_Survival, self).__getitem__(idx=idx)
        X = data_dict['Data']
        X = torch.reshape(X, (3, self.tile_size, self.tile_size))

        return {'Data': X,
                'Target': data_dict['Target'],
                'Censored': data_dict['Censored'],
                'Binary Target': data_dict['Binary Target'],
                'Time Target': data_dict['Time Target'],
                # 'Time List': data_dict['Time List'],
                'Time dict': data_dict['Time dict'],
                'File Names': data_dict['File Names'],
                'Images': data_dict['Images'],
                'Tile Locations': data_dict['Tile Locations'][0]
                }