import numpy as np
import os
import pandas as pd
import pickle
from random import sample
import torch
from torchvision import transforms
import sys
import time
from torch.utils.data import Dataset
from typing import List
from utils import MyRotation, Cutout, _get_tiles_2, _choose_data_3, _choose_data_2, chunks, HEDColorJitter
import matplotlib.pyplot as plt


MEAN = {'TCGA': [58.2069073 / 255, 96.22645279 / 255, 70.26442606 / 255],
        'HEROHE': [224.46091564 / 255, 190.67338568 / 255, 218.47883547 / 255],
        'Ron': [0.8998, 0.8253, 0.9357]
        }

STD = {'TCGA': [40.40400300279664 / 255, 58.90625962739444 / 255, 45.09334057330417 / 255],
       'HEROHE': [np.sqrt(1110.25292532) / 255, np.sqrt(2950.9804851) / 255, np.sqrt(1027.10911208) / 255],
       'Ron': [0.1125, 0.1751, 0.0787]
       }


class WSI_MILdataset(Dataset):
    def __init__(self,
                 DataSet: str = 'TCGA',
                 tile_size: int = 256,
                 bag_size: int = 50,
                 target_kind: str = 'ER',
                 test_fold: int = 1,
                 train: bool = True,
                 print_timing: bool = False,
                 transform : bool = False,
                 DX : bool = False,
                 get_images: bool = False):

        # Define data root:
        if sys.platform == 'linux':  # GIPdeep
            if (DataSet == 'HEROHE') or (DataSet == 'TCGA'):
                self.ROOT_PATH = r'/home/womer/project/All Data'
            elif DataSet == 'LUNG':
                self.ROOT_PATH = r'/home/rschley/All_Data/LUNG'
        elif sys.platform == 'win32':  # Ran local
            if DataSet == 'HEROHE':
                self.ROOT_PATH = r'C:\ran_data\HEROHE_examples'
            elif DataSet == 'TCGA':
                self.ROOT_PATH = r'C:\ran_data\TCGA_example_slides\TCGA_examples_131020_flat'
            elif DataSet == 'LUNG':
                self.ROOT_PATH = r'C:\ran_data\Lung_examples'
        else:  # Omer local
            if (DataSet == 'HEROHE') or (DataSet == 'TCGA'):
                self.ROOT_PATH = r'All Data'
            elif DataSet == 'LUNG':
                self.ROOT_PATH = 'All Data/LUNG'

        if DataSet == 'LUNG' and target_kind not in ['PDL1', 'EGFR']:
            raise ValueError('target should be one of: PDL1, EGFR')
        elif ((DataSet == 'HEROHE') or (DataSet == 'TCGA')) and target_kind not in ['ER', 'PR', 'Her2']:
            raise ValueError('target should be one of: ER, PR, Her2')

        if DataSet == 'RedSquares' or target_kind == 'RedSquares':
            meta_data_file = os.path.join(self.ROOT_PATH, 'slides_data_RedSquares.xlsx')
            DataSet = 'RedSquares'
            target_kind = 'RedSquares'
        else:
            meta_data_file = os.path.join(self.ROOT_PATH, 'slides_data.xlsx')

        self.DataSet = DataSet
        self.BASIC_MAGNIFICATION = 20
        if DataSet == 'RedSquares':
            self.BASIC_MAGNIFICATION = 10

        self.meta_data_DF = pd.read_excel(meta_data_file)

        if self.DataSet is not 'ALL':
            self.meta_data_DF = self.meta_data_DF[self.meta_data_DF['id'] == self.DataSet]
            self.meta_data_DF.reset_index(inplace=True)

        if DataSet == 'LUNG' and target_kind not in ['PDL1', 'EGFR']:
            raise ValueError('target should be one of: PDL1, EGFR')
        elif ((DataSet == 'HEROHE') or (DataSet == 'TCGA')) and target_kind not in ['ER', 'PR', 'Her2']:
            raise ValueError('target should be one of: ER, PR, Her2')
        if self.DataSet == 'LUNG':
            self.meta_data_DF = self.meta_data_DF[self.meta_data_DF['Origin'] == 'lung']
            self.meta_data_DF = self.meta_data_DF[self.meta_data_DF['Diagnosis'] == 'adenocarcinoma']
            self.meta_data_DF.reset_index(inplace=True)

        # self.meta_data_DF.set_index('id')
        ### self.data_path = os.path.join(self.ROOT_PATH, self.DataSet)
        self.tile_size = tile_size
        self.target_kind = target_kind
        self.test_fold = test_fold
        self.bag_size = bag_size
        self.train = train
        self.print_time = print_timing
        self.transform = transform
        self.DX = DX
        self.get_images = get_images

        all_targets = list(self.meta_data_DF[self.target_kind + ' status'])

        # We'll use only the valid slides - the ones with a Negative or Positive label.
        # Let's compute which slides are these:
        valid_slide_indices = np.where(np.isin(np.array(all_targets), ['Positive', 'Negative']) == True)[0]

        # Also remove slides without grid data:
        slides_without_grid = set(self.meta_data_DF.index[self.meta_data_DF['Total tiles - ' + str(self.tile_size) + ' compatible @ X20'] == -1])
        valid_slide_indices = np.array(list(set(valid_slide_indices) - slides_without_grid))

        # BUT...we want the train set to be a combination of all sets except the train set....Let's compute it:
        if self.train:
            folds = list(range(1, 7))
            folds.remove(self.test_fold)
        else:
            folds = [self.test_fold]
        self.folds = folds

        correct_folds = self.meta_data_DF['test fold idx'][valid_slide_indices].isin(folds)
        valid_slide_indices = np.array(correct_folds.index[correct_folds])

        all_image_file_names = list(self.meta_data_DF['file'])
        all_image_path_names = list(self.meta_data_DF['id'])
        all_in_fold = list(self.meta_data_DF['test fold idx'])
        all_tissue_tiles = list(self.meta_data_DF['Legitimate tiles - ' + str(self.tile_size) + ' compatible @ X20'])
        if self.DataSet is not 'TCGA':
            self.DX = False
        if self.DX:
            all_is_DX_cut = list(self.meta_data_DF['DX'])

        all_magnifications = list(self.meta_data_DF['Objective Power'])

        self.image_file_names = []
        self.image_path_names = []
        self.in_fold = []
        self.tissue_tiles = []
        self.target = []
        self.magnification = []

        for _, index in enumerate(valid_slide_indices):
            if (self.DX and all_is_DX_cut[index]) or not self.DX:
                self.image_file_names.append(all_image_file_names[index])
                self.image_path_names.append(all_image_path_names[index])
                self.in_fold.append(all_in_fold[index])
                self.tissue_tiles.append(all_tissue_tiles[index])
                self.target.append(all_targets[index])
                self.magnification.append(all_magnifications[index])

        # Setting the transformation:
        '''
        mean = {}
        std = {}

        mean['TCGA'] = [58.2069073 / 255, 96.22645279 / 255, 70.26442606 / 255]
        std['TCGA'] = [40.40400300279664 / 255, 58.90625962739444 / 255, 45.09334057330417 / 255]

        mean['HEROHE'] = [224.46091564 / 255, 190.67338568 / 255, 218.47883547 / 255]
        std['HEROHE'] = [np.sqrt(1110.25292532) / 255, np.sqrt(2950.9804851) / 255, np.sqrt(1027.10911208) / 255]

        mean['Ron'] = [0.8998, 0.8253, 0.9357]
        std['Ron'] = [0.1125, 0.1751, 0.0787]
        '''
        if self.transform and self.train:

            # TODO: Consider using - torchvision.transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
            # TODO: transforms.RandomRotation([self.rotate_by, self.rotate_by]),
            # TODO: Cutout(n_holes=1, length=100),
            # TODO: transforms.ColorJitter(brightness=(0.65, 1.35), contrast=(0.5, 1.5), saturation=(0.1), hue=(-0.1, 0.1)),
            # TODO: transforms.RandomHorizontalFlip(),

            self.transform = \
                transforms.Compose([ MyRotation(angles=[0, 90, 180, 270]),
                                     transforms.RandomVerticalFlip(),
                                     transforms.ToTensor(),
                                     #Cutout(n_holes=1, length=100),
                                     transforms.Normalize(
                                         mean=(MEAN['Ron'][0], MEAN['Ron'][1], MEAN['Ron'][2]),
                                         std=(STD['Ron'][0], STD['Ron'][1], STD['Ron'][2]))
                                     ])
            '''transforms.Normalize((0.8998, 0.8253, 0.9357), (0.1125, 0.1751, 0.0787))'''
            '''transforms.Normalize(
                mean=(MEAN[self.DataSet][0], MEAN[self.DataSet][1], MEAN[self.DataSet][2]),
                std=(STD[self.DataSet][0], STD[self.DataSet][1], STD[self.DataSet][2]))'''
        else:
            self.transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(
                                                     mean=(MEAN['Ron'][0], MEAN['Ron'][1], MEAN['Ron'][2]),
                                                     std=(STD['Ron'][0], STD['Ron'][1], STD['Ron'][2]))
                                                 ])
            '''transforms.Normalize((0.8998, 0.8253, 0.9357), (0.1125, 0.1751, 0.0787))'''
            '''transforms.Normalize(mean=(MEAN[self.DataSet][0], MEAN[self.DataSet][1], MEAN[self.DataSet][2]),
                                                     std=(STD[self.DataSet][0], STD[self.DataSet][1], STD[self.DataSet][2]))'''





        print('Initiation of WSI(MIL) {} {} DataSet for {} is Complete. {} Slides, Tiles of size {}^2. {} tiles in a bag, {} Transform. TestSet is fold #{}. DX is {}'
              .format('Train' if self.train else 'Test',
                      self.DataSet,
                      self.target_kind,
                      self.__len__(),
                      self.tile_size,
                      self.bag_size,
                      'Without' if transform is False else 'With',
                      self.test_fold,
                      'ON' if self.DX else 'OFF'))

    def __len__(self):
        return len(self.target)


    def __getitem__(self, idx):
        start_getitem = time.time()
        #data_path_extended = os.path.join(self.ROOT_PATH, self.image_path_names[idx])
        # file_name = os.path.join(self.data_path, self.image_path_names[idx], self.image_file_names[idx])
        # tiles = _choose_data(file_name, self.num_of_tiles_from_slide, self.magnification[idx], self.tile_size, print_timing=self.print_time)
        #tiles, time_list = _choose_data_2(self.data_path, file_name, self.bag_size, self.magnification[idx], self.tile_size, print_timing=self.print_time)
        '''
        tiles, time_list = _choose_data_2(data_path_extended, self.image_file_names[idx], self.bag_size, self.magnification[idx],
                                          self.tile_size, print_timing=self.print_time)
        '''
        basic_file_name = '.'.join(self.image_file_names[idx].split('.')[:-1])
        grid_file = os.path.join(self.ROOT_PATH, self.image_path_names[idx], 'Grids', basic_file_name + '--tlsz' + str(self.tile_size) + '.data')
        image_file = os.path.join(self.ROOT_PATH, self.image_path_names[idx], self.image_file_names[idx])

        tiles, time_list = _choose_data_2(grid_file, image_file, self.bag_size,
                                          self.magnification[idx],
                                          self.tile_size, print_timing=self.print_time)
        label = [1] if self.target[idx] == 'Positive' else [0]
        label = torch.LongTensor(label)

        # X will hold the images after all the transformations
        X = torch.zeros([self.bag_size, 3, self.tile_size, self.tile_size])

        '''
        # Updating RandomRotation angle in the data transformations only for train set:
        if self.train:
            rotate_by = sample([0, 90, 180, 270], 1)[0]
            transform = transforms.Compose([ transforms.RandomRotation([rotate_by, rotate_by]),
                                             self.transform
                                             ])
        else:
            transform = self.transform
        '''
        transform = self.transform


        magnification_relation = self.magnification[idx] // self.BASIC_MAGNIFICATION
        if magnification_relation != 1:
            transform = transforms.Compose([ transforms.Resize(self.tile_size), transform ])

        start_aug = time.time()
        for i in range(self.bag_size):
            X[i] = transform(tiles[i])
            '''
            img = get_concat(tiles[i], trans(X[i]))
            img.show()
            time.sleep(3)
            '''

        if self.get_images:
            images = torch.zeros([self.bag_size, 3, self.tile_size, self.tile_size])
            trans = transforms.ToTensor()
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


class Infer_WSI_MILdataset(Dataset):
    def __init__(self,
                 DataSet: str = 'HEROHE',
                 tile_size: int = 256,
                 tiles_per_iter: int = 400,
                 target_kind: str = 'ER',
                 folds: List = [1],
                 print_timing: bool = False,
                 DX: bool = False,
                 num_tiles: int = 500):

        self.ROOT_PATH = 'All Data'
        if DataSet == 'LUNG':
            self.ROOT_PATH = '/home/rschley/All_Data/LUNG'

        meta_data_file = os.path.join(self.ROOT_PATH, 'slides_data.xlsx')
        self.DataSet = DataSet
        self.BASIC_MAGNIFICATION = 20
        self.meta_data_DF = pd.read_excel(meta_data_file)
        if self.DataSet is not 'ALL':
            self.meta_data_DF = self.meta_data_DF[self.meta_data_DF['id'] == self.DataSet]
            self.meta_data_DF.reset_index(inplace=True)

        if DataSet == 'LUNG' and target_kind not in ['PDL1', 'EGFR']:
            raise ValueError('target should be one of: PDL1, EGFR')
        elif ((DataSet == 'HEROHE') or (DataSet == 'TCGA')) and target_kind not in ['ER', 'PR', 'Her2']:
            raise ValueError('target should be one of: ER, PR, Her2')
        if self.DataSet == 'HEROHE':
            target_kind = 'Her2'
        if self.DataSet == 'LUNG':
            self.meta_data_DF = self.meta_data_DF[self.meta_data_DF['Origin'] == 'lung']
            self.meta_data_DF = self.meta_data_DF[self.meta_data_DF['Diagnosis'] == 'adenocarcinoma']
            self.meta_data_DF.reset_index(inplace=True)

        self.tiles_per_iter = tiles_per_iter
        self.tile_size = tile_size
        self.target_kind = target_kind
        self.folds = folds
        self.print_time = print_timing
        self.DX = DX
        # The following attributes will be used in the __getitem__ function
        self.tiles_to_go = None
        self.slide_num = 0
        self.current_file = None

        all_targets = list(self.meta_data_DF[self.target_kind + ' status'])

        # We'll use only the valid slides - the ones with a Negative or Positive label.
        # Let's compute which slides are these:
        valid_slide_indices = np.where(np.isin(np.array(all_targets), ['Positive', 'Negative']) == True)[0]

        # Also remove slides without grid data:
        slides_without_grid = set(self.meta_data_DF.index[self.meta_data_DF['Total tiles - ' + str(self.tile_size) + ' compatible @ X20'] == -1])
        valid_slide_indices = np.array(list(set(valid_slide_indices) - slides_without_grid))

        correct_folds = self.meta_data_DF['test fold idx'][valid_slide_indices].isin(folds)
        valid_slide_indices = np.array(correct_folds.index[correct_folds])

        all_image_file_names = list(self.meta_data_DF['file'])
        all_image_path_names = list(self.meta_data_DF['id'])
        all_in_fold = list(self.meta_data_DF['test fold idx'])
        all_tissue_tiles = list(self.meta_data_DF['Legitimate tiles - ' + str(self.tile_size) + ' compatible @ X20'])
        if self.DataSet is not 'TCGA':
            self.DX = False
        if self.DX:
            all_is_DX_cut = list(self.meta_data_DF['DX'])

        all_magnifications = list(self.meta_data_DF['Objective Power'])

        self.image_file_names = []
        self.image_path_names = []
        self.in_fold = []
        self.tissue_tiles = []
        self.target = []
        self.magnification = []
        self.num_patches = []

        self.image_full_filenames = []
        self.slide_grids = []


        for _, slide_num in enumerate(valid_slide_indices):
            if (self.DX and all_is_DX_cut[slide_num]) or not self.DX:
                if num_tiles <= all_tissue_tiles[slide_num]:
                    self.num_patches.append(num_tiles)
                else:
                    self.num_patches.append(all_tissue_tiles[slide_num])
                    print('{} Slide available patches are less than {}'.format(all_image_file_names[slide_num], num_tiles))

                self.image_file_names.append(all_image_file_names[slide_num])
                self.image_path_names.append(all_image_path_names[slide_num])
                self.in_fold.append(all_in_fold[slide_num])
                self.tissue_tiles.append(all_tissue_tiles[slide_num])
                self.target.append(all_targets[slide_num])
                self.magnification.extend([all_magnifications[slide_num]] * self.num_patches[-1])

                full_image_filename = os.path.join(self.ROOT_PATH, self.image_path_names[-1], self.image_file_names[-1])
                self.image_full_filenames.append(full_image_filename)
                basic_file_name = '.'.join(self.image_file_names[-1].split('.')[:-1])
                grid_file = os.path.join(self.ROOT_PATH, self.image_path_names[-1], 'Grids',
                                         basic_file_name + '--tlsz' + str(self.tile_size) + '.data')
                which_patches = sample(range(int(self.tissue_tiles[-1])), self.num_patches[-1])

                with open(grid_file, 'rb') as filehandle:
                    grid_list = pickle.load(filehandle)
                chosen_locations = [ grid_list[loc] for loc in which_patches ]
                chosen_locations_chunks = chunks(chosen_locations, self.tiles_per_iter)
                self.slide_grids.extend(chosen_locations_chunks)
                ### self.slide_multiple_filenames.extend([full_image_filename] * self.num_patches[-1])

        # Setting the transformation:
        '''
        mean = {}
        std = {}

        mean['TCGA'] = [58.2069073 / 255, 96.22645279 / 255, 70.26442606 / 255]
        std['TCGA'] = [40.40400300279664 / 255, 58.90625962739444 / 255, 45.09334057330417 / 255]

        mean['HEROHE'] = [224.46091564 / 255, 190.67338568 / 255, 218.47883547 / 255]
        std['HEROHE'] = [np.sqrt(1110.25292532) / 255, np.sqrt(2950.9804851) / 255, np.sqrt(1027.10911208) / 255]

        mean['Ron'] = [0.8998, 0.8253, 0.9357]
        std['Ron'] = [0.1125, 0.1751, 0.0787]
        '''
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(
                                                 mean=(MEAN['Ron'][0], MEAN['Ron'][1], MEAN['Ron'][2]),
                                                 std=(STD['Ron'][0], STD['Ron'][1], STD['Ron'][2]))
                                             ])
        '''transforms.Normalize((0.8998, 0.8253, 0.9357), (0.1125, 0.1751, 0.0787))'''
        '''transforms.Normalize(mean=(MEAN[self.DataSet][0], MEAN[self.DataSet][1], MEAN[self.DataSet][2]),
                                                                  std=(STD[self.DataSet][0], STD[self.DataSet][1], STD[self.DataSet][2]))'''

        print("Normalization Values are:")
        print(MEAN['Ron'][0], MEAN['Ron'][1], MEAN['Ron'][2], STD['Ron'][0], STD['Ron'][1], STD['Ron'][2])
        print(
            'Initiation of WSI INFERENCE for {} DataSet and {} of folds {} is Complete. {} Slides, Working on Tiles of size {}^2. {} Tiles per slide, {} tiles per iteration, {} iterations to complete full inference'
            .format(self.DataSet,
                    self.target_kind,
                    str(self.folds),
                    len(self.image_file_names),
                    self.tile_size,
                    num_tiles,
                    self.tiles_per_iter,
                    self.__len__()))

    def __len__(self):
        ### return len(self.slide_multiple_filenames)
        return int(np.ceil(np.array(self.num_patches)/self.tiles_per_iter).sum())

    def __getitem__(self, idx):
        start_getitem = time.time()
        if self.tiles_to_go is None:
            self.tiles_to_go = self.num_patches[self.slide_num]
            self.current_file = self.image_full_filenames[self.slide_num]
            self.initial_num_patches = self.num_patches[self.slide_num]

        label = [1] if self.target[self.slide_num] == 'Positive' else [0]
        label = torch.LongTensor(label)

        if self.tiles_to_go <= self.tiles_per_iter:
            ### tiles_next_iter = self.tiles_to_go
            self.tiles_to_go = None
            self.slide_num += 1
        else:
            ### tiles_next_iter = self.tiles_per_iter
            self.tiles_to_go -= self.tiles_per_iter

        adjusted_tile_size = self.tile_size * (self.magnification[idx] // self.BASIC_MAGNIFICATION)
        tiles, time_list = _get_tiles_2(self.current_file,
                                       self.slide_grids[idx],
                                       adjusted_tile_size,
                                       self.print_time)

        X = torch.zeros([len(tiles), 3, self.tile_size, self.tile_size])

        magnification_relation = self.magnification[idx] // self.BASIC_MAGNIFICATION
        if magnification_relation != 1:
            transform = transforms.Compose([transforms.Resize(self.tile_size), self.transform])
        else:
            transform = self.transform

        start_aug = time.time()
        for i in range(len(tiles)):
            X[i] = transform(tiles[i])

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

        #print('Slide: {}, tiles: {}'.format(self.current_file, self.slide_grids[idx]))
        return X, label, time_list, last_batch, self.initial_num_patches


class WSI_REGdataset(Dataset):
    def __init__(self,
                 DataSet: str = 'TCGA',
                 tile_size: int = 256,
                 target_kind: str = 'ER',
                 test_fold: int = 1,
                 train: bool = True,
                 print_timing: bool = False,
                 DX: bool = False,
                 n_patches: int = 50,
                 transform_type: str = 'flip'):

        # Define data root:
        if sys.platform == 'linux': #GIPdeep
            if (DataSet == 'HEROHE') or (DataSet == 'TCGA'):
                self.ROOT_PATH = r'/home/womer/project/All Data'
            elif DataSet == 'LUNG':
                self.ROOT_PATH = r'/home/rschley/All_Data/LUNG'
        elif sys.platform == 'win32': #Ran local
            if DataSet == 'HEROHE':
                self.ROOT_PATH = r'C:\ran_data\HEROHE_examples'
            elif DataSet == 'TCGA':
                self.ROOT_PATH = r'C:\ran_data\TCGA_example_slides\TCGA_examples_131020_flat'
            elif DataSet == 'LUNG':
                self.ROOT_PATH = r'C:\ran_data\Lung_examples'
        else: #Omer local
            if (DataSet == 'HEROHE') or (DataSet == 'TCGA'):
                self.ROOT_PATH = r'All Data'
            elif DataSet == 'LUNG':
                self.ROOT_PATH = 'All Data/LUNG'

        if DataSet == 'LUNG' and target_kind not in ['PDL1', 'EGFR']:
            raise ValueError('target should be one of: PDL1, EGFR')
        elif ((DataSet == 'HEROHE') or (DataSet == 'TCGA')) and target_kind not in ['ER', 'PR', 'Her2']:
            raise ValueError('target should be one of: ER, PR, Her2')

        if DataSet == 'RedSquares' or target_kind == 'RedSquares':
            meta_data_file = os.path.join(self.ROOT_PATH, 'slides_data_RedSquares.xlsx')
            DataSet = 'RedSquares'
            target_kind = 'RedSquares'
        else:
            meta_data_file = os.path.join(self.ROOT_PATH, 'slides_data.xlsx')

        self.DataSet = DataSet
        self.BASIC_MAGNIFICATION = 20
        if DataSet == 'RedSquares':
            self.BASIC_MAGNIFICATION = 10

        self.meta_data_DF = pd.read_excel(meta_data_file)
        if self.DataSet is not 'ALL':
            self.meta_data_DF = self.meta_data_DF[self.meta_data_DF['id'] == self.DataSet]
            self.meta_data_DF.reset_index(inplace=True)

        #for lung, take only origin:lung and only diagnosis:adenocarcinoma
        if self.DataSet == 'LUNG':
            self.meta_data_DF = self.meta_data_DF[self.meta_data_DF['Origin'] == 'lung']
            self.meta_data_DF = self.meta_data_DF[self.meta_data_DF['Diagnosis'] == 'adenocarcinoma']
            self.meta_data_DF.reset_index(inplace=True)

        self.tile_size = tile_size
        self.target_kind = target_kind
        self.test_fold = test_fold
        self.bag_size = 1
        self.train = train
        self.print_time = print_timing
        self.DX = DX

        all_targets = list(self.meta_data_DF[self.target_kind + ' status'])

        # We'll use only the valid slides - the ones with a Negative or Positive label.
        # Let's compute which slides are these:
        valid_slide_indices = np.where(np.isin(np.array(all_targets), ['Positive', 'Negative']) == True)[0]

        # Also remove slides without grid data:
        slides_without_grid = set(self.meta_data_DF.index[self.meta_data_DF['Total tiles - ' + str(self.tile_size) + ' compatible @ X20'] == -1])
        valid_slide_indices = np.array(list(set(valid_slide_indices) - slides_without_grid))

        # BUT...we want the train set to be a combination of all sets except the train set....Let's compute it:
        if self.train:
            folds = list(self.meta_data_DF['test fold idx'].unique())
            folds.remove(self.test_fold)
            if 'test' in folds:
                folds.remove('test')
        else:
            folds = [self.test_fold]
        self.folds = folds

        correct_folds = self.meta_data_DF['test fold idx'][valid_slide_indices].isin(folds)
        valid_slide_indices = np.array(correct_folds.index[correct_folds])

        all_image_file_names = list(self.meta_data_DF['file'])
        all_image_path_names = list(self.meta_data_DF['id'])
        all_in_fold = list(self.meta_data_DF['test fold idx'])
        all_tissue_tiles = list(self.meta_data_DF['Legitimate tiles - ' + str(self.tile_size) + ' compatible @ X20'])
        if self.DataSet != 'TCGA':
            self.DX = False
        if self.DX:
            all_is_DX_cut = list(self.meta_data_DF['DX'])

        all_magnifications = list(self.meta_data_DF['Objective Power'])

        self.image_file_names = []
        self.image_path_names = []
        self.in_fold = []
        self.tissue_tiles = []
        self.target = []
        self.magnification = []

        for _, index in enumerate(valid_slide_indices):
            if (self.DX and all_is_DX_cut[index]) or not self.DX:
                self.image_file_names.append(all_image_file_names[index])
                self.image_path_names.append(all_image_path_names[index])
                self.in_fold.append(all_in_fold[index])
                self.tissue_tiles.append(all_tissue_tiles[index])
                self.target.append(all_targets[index])
                self.magnification.append(all_magnifications[index])

        # Setting the transformation:
        final_transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(
                                                mean=(MEAN['Ron'][0], MEAN['Ron'][1], MEAN['Ron'][2]),
                                                std=(STD['Ron'][0], STD['Ron'][1], STD['Ron'][2]))])
        #if self.transform and self.train:
        if transform_type != 'none' and self.train:
            # TODO: Consider using - torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
            # TODO: Consider using - torchvision.transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
            # TODO: Consider transforms.RandomHorizontalFlip()
            # TODO: transforms.RandomRotation([self.rotate_by, self.rotate_by]),
            if transform_type == 'flip':
                self.scale_factor = 0
                transform1 = \
                    transforms.Compose([ transforms.RandomVerticalFlip(),
                                         transforms.RandomHorizontalFlip()])
            elif transform_type == 'wcfrs': #weak color, flip, rotate, scale
                self.scale_factor = 0.2
                transform1 = \
                    transforms.Compose([
                        # transforms.ColorJitter(brightness=(0.65, 1.35), contrast=(0.5, 1.5),
                        transforms.ColorJitter(brightness=(0.85, 1.15), contrast=(0.75, 1.25),  # RanS 2.12.20
                                               saturation=0.1, hue=(-0.1, 0.1)),
                        transforms.RandomVerticalFlip(),
                        transforms.RandomHorizontalFlip(),
                        MyRotation(angles=[0, 90, 180, 270]),
                        transforms.RandomAffine(degrees=0, scale=(1 - self.scale_factor, 1 + self.scale_factor)),
                        #transforms.CenterCrop(self.tile_size),  #fix boundary when scaling<1
                        transforms.functional.crop(top=0, left=0, height=self.tile_size, width=self.tile_size)  # fix boundary when scaling<1
                    ])
            elif transform_type == 'hedcfrs':  # HED color, flip, rotate, scale
                self.scale_factor = 0.2
                transform1 = \
                    transforms.Compose([
                        transforms.ColorJitter(brightness=(0.85, 1.15), contrast=(0.75, 1.25)),
                        HEDColorJitter(sigma=0.05),
                        transforms.RandomVerticalFlip(),
                        transforms.RandomHorizontalFlip(),
                        MyRotation(angles=[0, 90, 180, 270]),
                        transforms.RandomAffine(degrees=0, scale=(1 - self.scale_factor, 1 + self.scale_factor)),
                        #transforms.CenterCrop(self.tile_size),  # fix boundary when scaling<1
                        transforms.functional.crop(top=0, left=0, height=self.tile_size, width=self.tile_size)  # fix boundary when scaling<1
                    ])


            self.transform = transforms.Compose([transform1,
                                                 final_transform])
        else:
            self.scale_factor = 0
            self.transform = final_transform

        self.factor = n_patches
        self.real_length = int(self.__len__() / self.factor)

        print('Initiation of REG {} {} DataSet for {} is Complete. {} Slides, Tiles of size {}^2. {} tiles in a bag, {} Transform. TestSet is fold #{}. DX is {}'
              .format('Train' if self.train else 'Test',
                      self.DataSet,
                      self.target_kind,
                      self.real_length,
                      self.tile_size,
                      self.bag_size,
                      #'Without' if transform is False else 'With',
                      'Without' if transform_type == 'none' else 'With',
                      self.test_fold,
                      'ON' if self.DX else 'OFF'))


    def __len__(self):
        return len(self.target) * self.factor


    def __getitem__(self, idx):
        start_getitem = time.time()
        idx = idx % self.real_length

        basic_file_name = '.'.join(self.image_file_names[idx].split('.')[:-1])
        grid_file = os.path.join(self.ROOT_PATH, self.image_path_names[idx], 'Grids', basic_file_name + '--tlsz' + str(self.tile_size) + '.data')
        image_file = os.path.join(self.ROOT_PATH, self.image_path_names[idx], self.image_file_names[idx])

        tiles, time_list = _choose_data_2(grid_file, image_file, self.bag_size,
                                          self.magnification[idx],
                                          #self.tile_size,
                                          int(self.tile_size / (1 - self.scale_factor)), # RanS 7.12.20, fix boundaries with scale
                                          print_timing=self.print_time)
        label = [1] if self.target[idx] == 'Positive' else [0]
        label = torch.LongTensor(label)

        # The following section is written for tiles in PIL format
        X = torch.zeros([1, 3, self.tile_size, self.tile_size])

        '''
        # Updating RandomRotation angle in the data transformations only for train set:
        if self.train:
            rotate_by = sample([0, 90, 180, 270], 1)[0]
            transform = transforms.Compose([ transforms.RandomRotation([rotate_by, rotate_by]),
                                             self.transform
                                             ])
        else:
            transform = self.transform
        '''
        transform = self.transform


        magnification_relation = self.magnification[idx] // self.BASIC_MAGNIFICATION
        if magnification_relation != 1:
            transform = transforms.Compose([ transforms.Resize(self.tile_size), transform ])

        start_aug = time.time()
        trans = transforms.ToPILImage()

        X = transform(tiles[0])


        temp = False
        if temp:
            trans1 = transforms.Compose([
                        #transforms.ColorJitter(brightness=(0.85, 1.15), contrast=(0.75, 1.25)),
                        HEDColorJitter(sigma=0.05)])
            trans2 = transforms.ColorJitter(saturation=0.1, hue=(-0.1, 0.1))

            colored_tile = trans1(tiles[0])
            colored_tile2 = trans2(tiles[0])
            fig, ax = plt.subplots(1,3)
            ax[0].imshow(tiles[0])
            ax[0].set_title('original')
            ax[1].imshow(colored_tile)
            ax[1].set_title('HED jitter')
            ax[2].imshow(colored_tile2)
            ax[2].set_title('RGB jitter')
            plt.show()
        '''
        img = get_concat(tiles[0], trans(X))
        img.show()
        time.sleep(3)
        '''

        aug_time = (time.time() - start_aug) / self.bag_size
        total_time = time.time() - start_getitem
        if self.print_time:
            time_list = (time_list[0], time_list[1], aug_time, total_time)
            # print('Data Augmentation time is: {:.2f} s'.format(aug_time))
            # print('WSI: TOTAL Time to prepare item is: {:.2f} s'.format(total_time))
        else:
            time_list = [0]

        slide_name = self.image_file_names[idx] #RanS 8.12.20
        return X, label, time_list, slide_name


class WSI_MIL2_dataset(Dataset):
    """
    This DataSet class is used for MIL paradigm training.
    This class uses patches from different slides (corresponding to the same label) is a bag.
    Half of the tiles will be taken from the main slide and all other tiles will be taken evenly from other slides
    """
    def __init__(self,
                 DataSet: str = 'TCGA',
                 tile_size: int = 256,
                 bag_size: int = 60,
                 TPS: int = 10, # Tiles Per Slide
                 target_kind: str = 'ER',
                 test_fold: int = 1,
                 train: bool = True,
                 print_timing: bool = False,
                 transform : bool = False,
                 DX : bool = False):

        self.ROOT_PATH = 'All Data'
        if DataSet == 'LUNG':
            self.ROOT_PATH = '/home/rschley/All_Data/LUNG'

        meta_data_file = os.path.join(self.ROOT_PATH, 'slides_data.xlsx')
        self.DataSet = DataSet
        self.BASIC_MAGNIFICATION = 20
        self.meta_data_DF = pd.read_excel(meta_data_file)
        if self.DataSet is not 'ALL':
            self.meta_data_DF = self.meta_data_DF[self.meta_data_DF['id'] == self.DataSet]
            self.meta_data_DF.reset_index(inplace=True)

        if DataSet == 'LUNG' and target_kind not in ['PDL1', 'EGFR']:
            raise ValueError('target should be one of: PDL1, EGFR')
        elif ((DataSet == 'HEROHE') or (DataSet == 'TCGA')) and target_kind not in ['ER', 'PR', 'Her2']:
            raise ValueError('target should be one of: ER, PR, Her2')
        if self.DataSet == 'HEROHE':
            target_kind = 'Her2'
        if self.DataSet == 'LUNG':
            self.meta_data_DF = self.meta_data_DF[self.meta_data_DF['Origin'] == 'lung']
            self.meta_data_DF = self.meta_data_DF[self.meta_data_DF['Diagnosis'] == 'adenocarcinoma']
            self.meta_data_DF.reset_index(inplace=True)

        # self.meta_data_DF.set_index('id')
        ### self.data_path = os.path.join(self.ROOT_PATH, self.DataSet)
        self.tile_size = tile_size
        self.target_kind = target_kind
        self.test_fold = test_fold
        self.bag_size = bag_size
        self.train = train
        self.print_time = print_timing
        self.transform = transform
        self.DX = DX

        # In test mode we want all tiles to be from the same slide:
        if self.train:
            self.TPS_original = int(self.bag_size / 2) # Half of bag will consist tiles from main slide
            self.TPS_others = TPS
        else:
            self.TPS_original = self.bag_size

        self.slides_in_bag = int((bag_size - self.TPS_original) / TPS)

        all_targets = list(self.meta_data_DF[self.target_kind + ' status'])

        # We'll use only the valid slides - the ones with a Negative or Positive label.
        # Let's compute which slides are these:
        valid_slide_indices = np.where(np.isin(np.array(all_targets), ['Positive', 'Negative']) == True)[0]

        # Also remove slides without grid data:
        slides_without_grid = set(self.meta_data_DF.index[self.meta_data_DF['Total tiles - ' + str(self.tile_size) + ' compatible @ X20'] == -1])
        valid_slide_indices = np.array(list(set(valid_slide_indices) - slides_without_grid))

        # BUT...we want the train set to be a combination of all sets except the train set....Let's compute it:
        if self.train:
            folds = list(range(1, 7))
            folds.remove(self.test_fold)
        else:
            folds = [self.test_fold]
        self.folds = folds

        correct_folds = self.meta_data_DF['test fold idx'][valid_slide_indices].isin(folds)
        valid_slide_indices = np.array(correct_folds.index[correct_folds])

        all_image_file_names = list(self.meta_data_DF['file'])
        all_image_path_names = list(self.meta_data_DF['id'])
        all_in_fold = list(self.meta_data_DF['test fold idx'])
        all_tissue_tiles = list(self.meta_data_DF['Legitimate tiles - ' + str(self.tile_size) + ' compatible @ X20'])
        if self.DataSet is not 'TCGA':
            self.DX = False
        if self.DX:
            all_is_DX_cut = list(self.meta_data_DF['DX'])

        all_magnifications = list(self.meta_data_DF['Objective Power'])

        self.image_file_names = []
        self.image_path_names = []
        self.in_fold = []
        self.tissue_tiles = []
        self.target = []
        self.magnification = []
        self.POS_slides, self.NEG_slides = [], []


        for i, index in enumerate(valid_slide_indices):
            if all_targets[index] == 'Positive':
                self.POS_slides.append(i)
            else:
                self.NEG_slides.append(i)

            if (self.DX and all_is_DX_cut[index]) or not self.DX:
                self.image_file_names.append(all_image_file_names[index])
                self.image_path_names.append(all_image_path_names[index])
                self.in_fold.append(all_in_fold[index])
                self.tissue_tiles.append(all_tissue_tiles[index])
                self.target.append(all_targets[index])
                self.magnification.append(all_magnifications[index])


        # Setting the transformation:
        '''
        mean = {}
        std = {}

        mean['TCGA'] = [58.2069073 / 255, 96.22645279 / 255, 70.26442606 / 255]
        std['TCGA'] = [40.40400300279664 / 255, 58.90625962739444 / 255, 45.09334057330417 / 255]

        mean['HEROHE'] = [224.46091564 / 255, 190.67338568 / 255, 218.47883547 / 255]
        std['HEROHE'] = [np.sqrt(1110.25292532) / 255, np.sqrt(2950.9804851) / 255, np.sqrt(1027.10911208) / 255]

        mean['Ron'] = [0.8998, 0.8253, 0.9357]
        std['Ron'] = [0.1125, 0.1751, 0.0787]
        '''
        if self.transform and self.train:

            # TODO: Consider using - torchvision.transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
            # TODO: transforms.RandomRotation([self.rotate_by, self.rotate_by]),
            # TODO: Cutout(n_holes=1, length=100),
            # TODO: transforms.ColorJitter(brightness=(0.65, 1.35), contrast=(0.5, 1.5), saturation=(0.1), hue=(-0.1, 0.1)),
            # TODO: transforms.RandomHorizontalFlip(),

            self.transform = \
                transforms.Compose([ transforms.RandomHorizontalFlip(),
                                     transforms.RandomVerticalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=(MEAN['Ron'][0], MEAN['Ron'][1], MEAN['Ron'][2]),
                                         std=(STD['Ron'][0], STD['Ron'][1], STD['Ron'][2]))
                                     ])
            '''transforms.Normalize((0.8998, 0.8253, 0.9357), (0.1125, 0.1751, 0.0787))'''
            '''transforms.Normalize(
                                         mean=(MEAN[self.DataSet][0], MEAN[self.DataSet][1], MEAN[self.DataSet][2]),
                                         std=(STD[self.DataSet][0], STD[self.DataSet][1], STD[self.DataSet][2]))'''
        else:
            self.transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(
                                                     mean=(MEAN['Ron'][0], MEAN['Ron'][1], MEAN['Ron'][2]),
                                                     std=(STD['Ron'][0], STD['Ron'][1], STD['Ron'][2]))
                                                 ])
            '''transforms.Normalize((0.8998, 0.8253, 0.9357), (0.1125, 0.1751, 0.0787))'''
            '''transforms.Normalize(
                                                     mean=(MEAN[self.DataSet][0], MEAN[self.DataSet][1], MEAN[self.DataSet][2]),
                                                     std=(STD[self.DataSet][0], STD[self.DataSet][1], STD[self.DataSet][2]))'''





        print('Initiation of WSI(MIL) {} {} DataSet for {} is Complete. {} Slides, Tiles of size {}^2. {} tiles in a bag, {} Transform. TestSet is fold #{}. DX is {}'
              .format('Train' if self.train else 'Test',
                      self.DataSet,
                      self.target_kind,
                      self.__len__(),
                      self.tile_size,
                      self.bag_size,
                      'Without' if transform is False else 'With',
                      self.test_fold,
                      'ON' if self.DX else 'OFF'))
        print('{} Tiles in a bag are gathered from: {} tiles from main slide + {} other slides'
              .format(self.bag_size, self.TPS_original, self.slides_in_bag))

    def __len__(self):
        return len(self.target)


    def __getitem__(self, idx):
        start_getitem = time.time()

        basic_file_name = '.'.join(self.image_file_names[idx].split('.')[:-1])
        grid_file = os.path.join(self.ROOT_PATH, self.image_path_names[idx], 'Grids', basic_file_name + '--tlsz' + str(self.tile_size) + '.data')
        image_file = os.path.join(self.ROOT_PATH, self.image_path_names[idx], self.image_file_names[idx])
        tiles, time_list = _choose_data_3(grid_file, image_file, self.TPS_original,
                                          self.magnification[idx],
                                          self.tile_size, print_timing=self.print_time)
        label = [1] if self.target[idx] == 'Positive' else [0]
        label = torch.LongTensor(label)

        # Sample tiles from other slides with same label - ONLY IN TRAIN MODE:
        if self.train:
            if self.target[idx] == 'Positive':
                slide_list = self.POS_slides
                slide_list.remove(idx)
                slides_idx_other = sample(slide_list, self.bag_size - 1)
            else:
                slide_list = self.NEG_slides
                slide_list.remove(idx)
                slides_idx_other = sample(slide_list, self.bag_size - 1)

        if self.train:
            for _, index in enumerate(slides_idx_other):
                basic_file_name_other = '.'.join(self.image_file_names[index].split('.')[:-1])
                grid_file_other = os.path.join(self.ROOT_PATH, self.image_path_names[index], 'Grids',
                                           basic_file_name_other + '--tlsz' + str(self.tile_size) + '.data')
                image_file_other = os.path.join(self.ROOT_PATH, self.image_path_names[index], self.image_file_names[index])

                tiles_other, time_list_other = _choose_data_3(grid_file_other, image_file_other, self.TPS_others,
                                                          self.magnification[index],
                                                          self.tile_size, print_timing=self.print_time)
                tiles.extend(tiles_other)

        X = torch.zeros([self.bag_size, 3, self.tile_size, self.tile_size])

        transform = self.transform
        '''
        magnification_relation = self.magnification[idx] // self.BASIC_MAGNIFICATION
        if magnification_relation != 1:
            transform = transforms.Compose([ transforms.Resize(self.tile_size), transform ])
        '''
        start_aug = time.time()
        ### trans = transforms.ToPILImage()
        for i in range(self.bag_size):
            X[i] = transform(tiles[i])

        aug_time = (time.time() - start_aug) / self.bag_size
        total_time = time.time() - start_getitem
        if self.print_time:
            time_list = (time_list[0], time_list[1], aug_time, total_time)
        else:
            time_list = [0]

        trans = transforms.ToTensor()
        return X, label, time_list, self.image_file_names[idx], trans(tiles[0])


class WSI_MIL3_dataset(Dataset):
    """
    This DataSet class is used for MIL paradigm training.
    This class uses patches from different slides (corresponding to the same label) is a bag.
    It will use equall amount of tiles from each slide
    """
    def __init__(self,
                 DataSet: str = 'TCGA',
                 tile_size: int = 256,
                 bag_size: int = 50,
                 TPS: int = 10, # Tiles Per Slide
                 target_kind: str = 'ER',
                 test_fold: int = 1,
                 train: bool = True,
                 print_timing: bool = False,
                 transform : bool = False,
                 DX : bool = False):

        self.ROOT_PATH = 'All Data'
        if DataSet == 'LUNG':
            self.ROOT_PATH = '/home/rschley/All_Data/LUNG'

        meta_data_file = os.path.join(self.ROOT_PATH, 'slides_data.xlsx')
        self.DataSet = DataSet
        self.BASIC_MAGNIFICATION = 20
        self.meta_data_DF = pd.read_excel(meta_data_file)
        if self.DataSet is not 'ALL':
            self.meta_data_DF = self.meta_data_DF[self.meta_data_DF['id'] == self.DataSet]
            self.meta_data_DF.reset_index(inplace=True)

        if DataSet == 'LUNG' and target_kind not in ['PDL1', 'EGFR']:
            raise ValueError('target should be one of: PDL1, EGFR')
        elif ((DataSet == 'HEROHE') or (DataSet == 'TCGA')) and target_kind not in ['ER', 'PR', 'Her2']:
            raise ValueError('target should be one of: ER, PR, Her2')
        if self.DataSet == 'HEROHE':
            target_kind = 'Her2'
        if self.DataSet == 'LUNG':
            self.meta_data_DF = self.meta_data_DF[self.meta_data_DF['Origin'] == 'lung']
            self.meta_data_DF = self.meta_data_DF[self.meta_data_DF['Diagnosis'] == 'adenocarcinoma']
            self.meta_data_DF.reset_index(inplace=True)

        self.tile_size = tile_size
        self.target_kind = target_kind
        self.test_fold = test_fold
        self.bag_size = bag_size
        self.train = train
        self.print_time = print_timing
        self.transform = transform
        self.DX = DX

        # In test mode we want all tiles to be from the same slide.
        # in train mode all tiles will be taken evenly from slides with same label
        if self.train:
            self.TPS = TPS
        else:
            self.TPS = self.bag_size

        self.slides_in_bag = int(bag_size / self.TPS)

        all_targets = list(self.meta_data_DF[self.target_kind + ' status'])

        # We'll use only the valid slides - the ones with a Negative or Positive label.
        # Let's compute which slides are these:
        valid_slide_indices = np.where(np.isin(np.array(all_targets), ['Positive', 'Negative']) == True)[0]

        # Also remove slides without grid data:
        slides_without_grid = set(self.meta_data_DF.index[self.meta_data_DF['Total tiles - ' + str(self.tile_size) + ' compatible @ X20'] == -1])
        valid_slide_indices = np.array(list(set(valid_slide_indices) - slides_without_grid))

        # BUT...we want the train set to be a combination of all sets except the train set....Let's compute it:
        if self.train:
            folds = list(range(1, 7))
            folds.remove(self.test_fold)
        else:
            folds = [self.test_fold]
        self.folds = folds

        correct_folds = self.meta_data_DF['test fold idx'][valid_slide_indices].isin(folds)
        valid_slide_indices = np.array(correct_folds.index[correct_folds])

        all_image_file_names = list(self.meta_data_DF['file'])
        all_image_path_names = list(self.meta_data_DF['id'])
        all_in_fold = list(self.meta_data_DF['test fold idx'])
        all_tissue_tiles = list(self.meta_data_DF['Legitimate tiles - ' + str(self.tile_size) + ' compatible @ X20'])
        if self.DataSet is not 'TCGA':
            self.DX = False
        if self.DX:
            all_is_DX_cut = list(self.meta_data_DF['DX'])

        all_magnifications = list(self.meta_data_DF['Objective Power'])

        self.image_file_names = []
        self.image_path_names = []
        self.in_fold = []
        self.tissue_tiles = []
        self.target = []
        self.magnification = []
        self.POS_slides, self.NEG_slides = [], []


        for i, index in enumerate(valid_slide_indices):
            if all_targets[index] == 'Positive':
                self.POS_slides.append(i)
            else:
                self.NEG_slides.append(i)

            if (self.DX and all_is_DX_cut[index]) or not self.DX:
                self.image_file_names.append(all_image_file_names[index])
                self.image_path_names.append(all_image_path_names[index])
                self.in_fold.append(all_in_fold[index])
                self.tissue_tiles.append(all_tissue_tiles[index])
                self.target.append(all_targets[index])
                self.magnification.append(all_magnifications[index])


        # Setting the transformation:
        '''
        mean = {}
        std = {}

        mean['TCGA'] = [58.2069073 / 255, 96.22645279 / 255, 70.26442606 / 255]
        std['TCGA'] = [40.40400300279664 / 255, 58.90625962739444 / 255, 45.09334057330417 / 255]

        mean['HEROHE'] = [224.46091564 / 255, 190.67338568 / 255, 218.47883547 / 255]
        std['HEROHE'] = [np.sqrt(1110.25292532) / 255, np.sqrt(2950.9804851) / 255, np.sqrt(1027.10911208) / 255]

        mean['Ron'] = [0.8998, 0.8253, 0.9357]
        std['Ron'] = [0.1125, 0.1751, 0.0787]
        '''
        if self.transform and self.train:

            # TODO: Consider using - torchvision.transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
            # TODO: transforms.RandomRotation([self.rotate_by, self.rotate_by]),
            # TODO: Cutout(n_holes=1, length=100),
            # TODO: transforms.ColorJitter(brightness=(0.65, 1.35), contrast=(0.5, 1.5), saturation=(0.1), hue=(-0.1, 0.1)),
            # TODO: transforms.RandomHorizontalFlip(),

            self.transform = \
                transforms.Compose([ transforms.RandomHorizontalFlip(),
                                     transforms.RandomVerticalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=(MEAN['Ron'][0], MEAN['Ron'][1], MEAN['Ron'][2]),
                                         std=(STD['Ron'][0], STD['Ron'][1], STD['Ron'][2]))
                                     ])
            '''transforms.Normalize((0.8998, 0.8253, 0.9357), (0.1125, 0.1751, 0.0787))'''
            '''transforms.Normalize(
                                         mean=(MEAN[self.DataSet][0], MEAN[self.DataSet][1], MEAN[self.DataSet][2]),
                                         std=(STD[self.DataSet][0], STD[self.DataSet][1], STD[self.DataSet][2]))'''
        else:
            self.transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(
                                                     mean=(MEAN['Ron'][0], MEAN['Ron'][1], MEAN['Ron'][2]),
                                                     std=(STD['Ron'][0], STD['Ron'][1], STD['Ron'][2]))
                                                 ])
            '''transforms.Normalize((0.8998, 0.8253, 0.9357), (0.1125, 0.1751, 0.0787))'''
            '''transforms.Normalize(
                                                     mean=(MEAN[self.DataSet][0], MEAN[self.DataSet][1], MEAN[self.DataSet][2]),
                                                     std=(STD[self.DataSet][0], STD[self.DataSet][1], STD[self.DataSet][2]))'''





        print('Initiation of WSI(MIL) {} {} DataSet for {} is Complete. {} Slides, Tiles of size {}^2. {} tiles in a bag, {} Transform. TestSet is fold #{}. DX is {}'
              .format('Train' if self.train else 'Test',
                      self.DataSet,
                      self.target_kind,
                      self.__len__(),
                      self.tile_size,
                      self.bag_size,
                      'Without' if transform is False else 'With',
                      self.test_fold,
                      'ON' if self.DX else 'OFF'))
        print('{} Tiles in a bag are gathered EVENLY from {} slides'
              .format(self.bag_size, self.slides_in_bag))

    def __len__(self):
        return len(self.target)


    def __getitem__(self, idx):
        start_getitem = time.time()

        basic_file_name = '.'.join(self.image_file_names[idx].split('.')[:-1])
        grid_file = os.path.join(self.ROOT_PATH, self.image_path_names[idx], 'Grids', basic_file_name + '--tlsz' + str(self.tile_size) + '.data')
        image_file = os.path.join(self.ROOT_PATH, self.image_path_names[idx], self.image_file_names[idx])
        tiles, time_list = _choose_data_3(grid_file, image_file, self.TPS,
                                          self.magnification[idx],
                                          self.tile_size, print_timing=self.print_time)
        label = [1] if self.target[idx] == 'Positive' else [0]
        label = torch.LongTensor(label)

        # Sample tiles from other slides with same label - ONLY IN TRAIN MODE:
        if self.train:
            if self.target[idx] == 'Positive':
                slide_list = self.POS_slides
                slide_list.remove(idx)
                slides_idx_other = sample(slide_list, self.bag_size - 1)
            else:
                slide_list = self.NEG_slides
                slide_list.remove(idx)
                slides_idx_other = sample(slide_list, self.bag_size - 1)

        if self.train:
            for _, index in enumerate(slides_idx_other):
                basic_file_name_other = '.'.join(self.image_file_names[index].split('.')[:-1])
                grid_file_other = os.path.join(self.ROOT_PATH, self.image_path_names[index], 'Grids',
                                           basic_file_name_other + '--tlsz' + str(self.tile_size) + '.data')
                image_file_other = os.path.join(self.ROOT_PATH, self.image_path_names[index], self.image_file_names[index])

                tiles_other, time_list_other = _choose_data_3(grid_file_other, image_file_other, self.TPS,
                                                              self.magnification[index],
                                                              self.tile_size, print_timing=self.print_time)
                tiles.extend(tiles_other)

        X = torch.zeros([self.bag_size, 3, self.tile_size, self.tile_size])

        transform = self.transform
        '''
        magnification_relation = self.magnification[idx] // self.BASIC_MAGNIFICATION
        if magnification_relation != 1:
            transform = transforms.Compose([ transforms.Resize(self.tile_size), transform ])
        '''
        start_aug = time.time()
        ### trans = transforms.ToPILImage()
        for i in range(self.bag_size):
            X[i] = transform(tiles[i])

        aug_time = (time.time() - start_aug) / self.bag_size
        total_time = time.time() - start_getitem
        if self.print_time:
            time_list = (time_list[0], time_list[1], aug_time, total_time)
        else:
            time_list = [0]

        trans = transforms.ToTensor()
        return X, label, time_list, self.image_file_names[idx], trans(tiles[0])


class WSI_MIL_OFTest_dataset(Dataset):
    def __init__(self,
                 DataSet: str = 'LUNG',
                 tile_size: int = 256,
                 bag_size: int = 50,
                 target_kind: str = 'PDL1',
                 test_fold: int = 5,
                 train: bool = True,
                 print_timing: bool = False,
                 transform : bool = True,
                 DX : bool = False,
                 get_images: bool = False):

        self.ROOT_PATH = 'All Data'
        if DataSet == 'LUNG':
            self.ROOT_PATH = '/home/rschley/All_Data/LUNG'

        meta_data_file = os.path.join(self.ROOT_PATH, 'slides_data.xlsx')
        if sys.platform != 'linux':
            self.ROOT_PATH = 'All Data/LUNG'
            meta_data_file = os.path.join('All Data/slides_data_LUNG.xlsx')

        self.DataSet = DataSet
        self.BASIC_MAGNIFICATION = 20
        self.meta_data_DF = pd.read_excel(meta_data_file)
        if self.DataSet is not 'ALL':
            self.meta_data_DF = self.meta_data_DF[self.meta_data_DF['id'] == self.DataSet]
            self.meta_data_DF.reset_index(inplace=True)

        if DataSet == 'LUNG' and target_kind not in ['PDL1', 'EGFR']:
            raise ValueError('target should be one of: PDL1, EGFR')
        elif ((DataSet == 'HEROHE') or (DataSet == 'TCGA')) and target_kind not in ['ER', 'PR', 'Her2']:
            raise ValueError('target should be one of: ER, PR, Her2')
        if self.DataSet == 'HEROHE':
            target_kind = 'Her2'
        if self.DataSet == 'LUNG':
            self.meta_data_DF = self.meta_data_DF[self.meta_data_DF['Origin'] == 'lung']
            self.meta_data_DF = self.meta_data_DF[self.meta_data_DF['Diagnosis'] == 'adenocarcinoma']
            self.meta_data_DF.reset_index(inplace=True)

        # self.meta_data_DF.set_index('id')
        ### self.data_path = os.path.join(self.ROOT_PATH, self.DataSet)
        self.tile_size = tile_size
        self.target_kind = target_kind
        self.test_fold = test_fold
        self.bag_size = bag_size
        self.train = train
        self.print_time = print_timing
        self.transform = transform
        self.DX = DX
        self.get_images = get_images

        all_targets = list(self.meta_data_DF[self.target_kind + ' status'])

        # We'll use only the valid slides - the ones with a Negative or Positive label.
        # Let's compute which slides are these:
        valid_slide_indices = np.where(np.isin(np.array(all_targets), ['Positive', 'Negative']) == True)[0]

        all_slide_names = list(self.meta_data_DF['patient barcode'])

        slides_in_train = ['1076-17 HE',  # POSITIVE
                           '12607-17',
                           '11207-17',  # NEGATIVE
                           '12815-17']

        valid_slide_indices_OFTest = np.where(np.isin(np.array(all_slide_names), slides_in_train) == True)[0]

        # Also remove slides without grid data:
        slides_without_grid = set(self.meta_data_DF.index[self.meta_data_DF['Total tiles - ' + str(self.tile_size) + ' compatible @ X20'] == -1])
        valid_slide_indices = np.array(list(set(valid_slide_indices_OFTest) - slides_without_grid))

        # BUT...we want the train set to be a combination of all sets except the train set....Let's compute it:
        if self.train:
            folds = list(range(1, 7))
            folds.remove(self.test_fold)
        else:
            folds = [self.test_fold]
        self.folds = folds

        correct_folds = self.meta_data_DF['test fold idx'][valid_slide_indices].isin(folds)
        valid_slide_indices = np.array(correct_folds.index[correct_folds])

        all_image_file_names = list(self.meta_data_DF['file'])
        all_image_path_names = list(self.meta_data_DF['id'])
        all_in_fold = list(self.meta_data_DF['test fold idx'])
        all_tissue_tiles = list(self.meta_data_DF['Legitimate tiles - ' + str(self.tile_size) + ' compatible @ X20'])
        if self.DataSet is not 'TCGA':
            self.DX = False
        if self.DX:
            all_is_DX_cut = list(self.meta_data_DF['DX'])

        all_magnifications = list(self.meta_data_DF['Objective Power'])

        self.image_file_names = []
        self.image_path_names = []
        self.in_fold = []
        self.tissue_tiles = []
        self.target = []
        self.magnification = []

        for _, index in enumerate(valid_slide_indices):
            if (self.DX and all_is_DX_cut[index]) or not self.DX:
                self.image_file_names.append(all_image_file_names[index])
                self.image_path_names.append(all_image_path_names[index])
                self.in_fold.append(all_in_fold[index])
                self.tissue_tiles.append(all_tissue_tiles[index])
                self.target.append(all_targets[index])
                self.magnification.append(all_magnifications[index])

        # Setting the transformation:
        '''
        mean = {}
        std = {}

        mean['TCGA'] = [58.2069073 / 255, 96.22645279 / 255, 70.26442606 / 255]
        std['TCGA'] = [40.40400300279664 / 255, 58.90625962739444 / 255, 45.09334057330417 / 255]

        mean['HEROHE'] = [224.46091564 / 255, 190.67338568 / 255, 218.47883547 / 255]
        std['HEROHE'] = [np.sqrt(1110.25292532) / 255, np.sqrt(2950.9804851) / 255, np.sqrt(1027.10911208) / 255]

        mean['Ron'] = [0.8998, 0.8253, 0.9357]
        std['Ron'] = [0.1125, 0.1751, 0.0787]
        '''
        if self.transform and self.train:

            # TODO: Consider using - torchvision.transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
            # TODO: transforms.RandomRotation([self.rotate_by, self.rotate_by]),
            # TODO: Cutout(n_holes=1, length=100),
            # TODO: transforms.ColorJitter(brightness=(0.65, 1.35), contrast=(0.5, 1.5), saturation=(0.1), hue=(-0.1, 0.1)),
            # TODO: transforms.RandomHorizontalFlip(),

            self.transform = \
                transforms.Compose([ MyRotation(angles=[0, 90, 180, 270]),
                                     transforms.RandomVerticalFlip(),
                                     transforms.ToTensor(),
                                     #Cutout(n_holes=1, length=100),
                                     transforms.Normalize(
                                         mean=(MEAN['Ron'][0], MEAN['Ron'][1], MEAN['Ron'][2]),
                                         std=(STD['Ron'][0], STD['Ron'][1], STD['Ron'][2]))
                                     ])
            '''transforms.Normalize((0.8998, 0.8253, 0.9357), (0.1125, 0.1751, 0.0787))'''
            '''transforms.Normalize(
                mean=(MEAN[self.DataSet][0], MEAN[self.DataSet][1], MEAN[self.DataSet][2]),
                std=(STD[self.DataSet][0], STD[self.DataSet][1], STD[self.DataSet][2]))'''
        else:
            self.transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(
                                                     mean=(MEAN['Ron'][0], MEAN['Ron'][1], MEAN['Ron'][2]),
                                                     std=(STD['Ron'][0], STD['Ron'][1], STD['Ron'][2]))
                                                 ])
            '''transforms.Normalize((0.8998, 0.8253, 0.9357), (0.1125, 0.1751, 0.0787))'''
            '''transforms.Normalize(mean=(MEAN[self.DataSet][0], MEAN[self.DataSet][1], MEAN[self.DataSet][2]),
                                                     std=(STD[self.DataSet][0], STD[self.DataSet][1], STD[self.DataSet][2]))'''





        print('Initiation of WSI(MIL) {} {} DataSet for {} is Complete. {} Slides, Tiles of size {}^2. {} tiles in a bag, {} Transform. TestSet is fold #{}. DX is {}'
              .format('Train' if self.train else 'Test',
                      self.DataSet,
                      self.target_kind,
                      self.__len__(),
                      self.tile_size,
                      self.bag_size,
                      'Without' if transform is False else 'With',
                      self.test_fold,
                      'ON' if self.DX else 'OFF'))

    def __len__(self):
        return len(self.target)


    def __getitem__(self, idx):
        start_getitem = time.time()
        #data_path_extended = os.path.join(self.ROOT_PATH, self.image_path_names[idx])
        # file_name = os.path.join(self.data_path, self.image_path_names[idx], self.image_file_names[idx])
        # tiles = _choose_data(file_name, self.num_of_tiles_from_slide, self.magnification[idx], self.tile_size, print_timing=self.print_time)
        #tiles, time_list = _choose_data_2(self.data_path, file_name, self.bag_size, self.magnification[idx], self.tile_size, print_timing=self.print_time)
        '''
        tiles, time_list = _choose_data_2(data_path_extended, self.image_file_names[idx], self.bag_size, self.magnification[idx],
                                          self.tile_size, print_timing=self.print_time)
        '''
        basic_file_name = '.'.join(self.image_file_names[idx].split('.')[:-1])
        grid_file = os.path.join(self.ROOT_PATH, self.image_path_names[idx], 'Grids', basic_file_name + '--tlsz' + str(self.tile_size) + '.data')
        image_file = os.path.join(self.ROOT_PATH, self.image_path_names[idx], self.image_file_names[idx])

        tiles, time_list = _choose_data_2(grid_file, image_file, self.bag_size,
                                          self.magnification[idx],
                                          self.tile_size, print_timing=self.print_time)
        label = [1] if self.target[idx] == 'Positive' else [0]
        label = torch.LongTensor(label)

        # X will hold the images after all the transformations
        X = torch.zeros([self.bag_size, 3, self.tile_size, self.tile_size])

        '''
        # Updating RandomRotation angle in the data transformations only for train set:
        if self.train:
            rotate_by = sample([0, 90, 180, 270], 1)[0]
            transform = transforms.Compose([ transforms.RandomRotation([rotate_by, rotate_by]),
                                             self.transform
                                             ])
        else:
            transform = self.transform
        '''
        transform = self.transform


        magnification_relation = self.magnification[idx] // self.BASIC_MAGNIFICATION
        if magnification_relation != 1:
            transform = transforms.Compose([ transforms.Resize(self.tile_size), transform ])

        start_aug = time.time()
        for i in range(self.bag_size):
            X[i] = transform(tiles[i])
            '''
            img = get_concat(tiles[i], trans(X[i]))
            img.show()
            time.sleep(3)
            '''

        if self.get_images:
            images = torch.zeros([self.bag_size, 3, self.tile_size, self.tile_size])
            trans = transforms.ToTensor()
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