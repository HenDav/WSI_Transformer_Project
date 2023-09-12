import logging
import os
import sys
from pathlib import Path
from random import shuffle
from typing import List

import numpy as np
import torch
from PIL import Image

from utils import run_data


def run_data_multi_model(experiments: List[str] = None, models: List[str] = None,
                         epoch: int = None, transformation_string: str = None):
    num_experiments = len(experiments)
    if experiments is not None and transformation_string is not None:
        for index in range(num_experiments):
            run_data(experiment=experiments[index], transformation_string=transformation_string)
    elif experiments is not None and models is not None:
        for index in range(num_experiments):
            run_data(experiment=experiments[index], model=models[index])
    elif experiments is not None and epoch is not None:
        for index in range(num_experiments):
            run_data(experiment=experiments[index], epoch=epoch)


def gather_per_patient_data(all_targets, all_scores_for_class_1, all_patient_barcodes):
    """
    This function gets 3 lists containing data about slides targets, scores (for class 1 - positive) and patient barcodes.
    The function computes and returns the mean score for all slides that belong to the same patient.
    The function uses the targets list to make sure that all targets for the same patient are the equal and return it's value

    :param all_targets:
    :param all_scores_for_class_1:
    :param all_patient_barcodes:
    :return:
    """

    targets_dict = {}
    scores_dict = {}

    # first, we'll gather all the data for specific patients
    for idx, patient in enumerate(all_patient_barcodes):
        if patient in targets_dict:
            targets_dict[patient].append(all_targets[idx])
            scores_dict[patient].append(all_scores_for_class_1[idx])
        else:
            targets_dict[patient] = [all_targets[idx]]
            scores_dict[patient] = [all_scores_for_class_1[idx]]

    # Now, we'll find the mean values for each patient:
    all_targets_per_patient, all_scores_for_class_1_per_patient = [], []
    patient_barcodes = targets_dict.keys()
    for barcode in patient_barcodes:
        targets = np.array(targets_dict[barcode])
        scores_mean = np.array(scores_dict[barcode]).mean()

        # Check that all targets for the same patient are the same.
        if targets[0] != targets.mean():
            raise Exception('Not all targets for patient {} are equal'.format(barcode))

        all_targets_per_patient.append(int(targets.mean()))
        all_scores_for_class_1_per_patient.append(scores_mean)

    return all_targets_per_patient, all_scores_for_class_1_per_patient


def heb2rgb(imageHEB, HEBtoRGB_Conversion_Matrix):
    # Dividing channels:
    """
    image_H = np.zeros_like(imageHEB)
    image_H[:, :, 0] = imageHEB[:, :, 0]
    image_H_flattened = np.stack([image_H[:, :, 0].ravel(), image_H[:, :, 1].ravel(), image_H[:, :, 2].ravel()], axis=0)
    """

    image_HEB_flattened = np.stack([imageHEB[:, :, 0].ravel(), imageHEB[:, :, 1].ravel(), imageHEB[:, :, 2].ravel()],
                                   axis=0)
    # convert to RGB:
    Optical_Density_flattened = np.matmul(HEBtoRGB_Conversion_Matrix, image_HEB_flattened)

    # unflatten:
    OD = np.zeros_like(imageHEB)
    OD[:, :, 0] = Optical_Density_flattened[0].reshape(imageHEB.shape[0], imageHEB.shape[1])
    OD[:, :, 1] = Optical_Density_flattened[1].reshape(imageHEB.shape[0], imageHEB.shape[1])
    OD[:, :, 2] = Optical_Density_flattened[2].reshape(imageHEB.shape[0], imageHEB.shape[1])

    # exp10 to go back from Optical Density to Intensity:
    image_RGB = np.power(10, -OD)

    # Clip the image to have values in the range 0-1:
    image_RGB = np.clip(image_RGB, 0, 1)

    return image_RGB


def rgb2heb(imageRGB, color_values: list = None):
    """
    takes an RGB image of a slide stained by H&E,
    separates the colors into the components of the stains
    returns the imageHEB which containts H, E, and background images
    """

    if type(imageRGB) == Image.Image:
        imageRGB = np.asarray(imageRGB)

    # remove zeros by scaling
    imageRGB = (imageRGB.astype(np.uint16) + 1) / 256

    if color_values is None:
        # HE values from paper:
        He = np.array([0.18, 0.2, 0.08])
        Eo = np.array([0.01, 0.13, 0.01])
        Res = np.array([0.1, 0.21, 0.29])  # DAB

    else:
        He = -np.log10(np.array(color_values[0]) / 255)
        Eo = -np.log10(np.array(color_values[1]) / 255)
        Res = -np.log10(np.array(color_values[2]) / 255)

    # combine stain vectors to deconvolution matrix
    HEtoRGB = np.stack([He / np.linalg.norm(He), Eo / np.linalg.norm(Eo), Res / np.linalg.norm(Res)], axis=1)
    RGBtoHE = np.linalg.inv(HEtoRGB)

    # perform color deconvolution
    image_rgb_flattened = np.stack([imageRGB[:, :, 0].ravel(), imageRGB[:, :, 1].ravel(), imageRGB[:, :, 2].ravel()],
                                   axis=0)
    try:
        imageHEB_flattened = np.matmul(RGBtoHE, -np.log10(image_rgb_flattened))
    except RuntimeWarning:
        logging.info('DEBUG ME !!!!')

    # Converting back to 3 channels:
    imageHEB = np.zeros_like(imageRGB)
    imageHEB[:, :, 0] = imageHEB_flattened[0].reshape(imageRGB.shape[0], imageRGB.shape[1])
    imageHEB[:, :, 1] = imageHEB_flattened[1].reshape(imageRGB.shape[0], imageRGB.shape[1])
    imageHEB[:, :, 2] = imageHEB_flattened[2].reshape(imageRGB.shape[0], imageRGB.shape[1])

    return imageHEB, HEtoRGB,


def concatenate_minibatch(minibatch, is_shuffle: bool = False):
    if minibatch[1] == 0:  # There is 1 combined dataset that includes all samples
        return minibatch[0]

    else:  # There are 2 dataset, one for Censored samples and one for not censored samples. Those 2 datasets need to be combined
        # The data in the combined dataset is divided into censored and not censored.
        #  We'll shuffle it:
        indices = list(range(minibatch[0]['Censored'].size(0) + minibatch[1]['Censored'].size(0)))
        if is_shuffle:
            shuffle(indices)

        temp_minibatch = {}
        for key in minibatch[0].keys():
            if type(minibatch[0][key]) == torch.Tensor:
                temp_minibatch[key] = torch.cat([minibatch[0][key], minibatch[1][key]], dim=0)[indices]

            elif type(minibatch[0][key]) == list:
                if key == 'Tile Locations':
                    x_locs = minibatch[0][key][0].tolist() + minibatch[1][key][0].tolist()
                    y_locs = minibatch[0][key][1].tolist() + minibatch[1][key][1].tolist()
                    temp_minibatch[key] = {'X': x_locs,
                                           'Y': y_locs
                                           }

                else:
                    minibatch[0][key].extend(minibatch[1][key])
                    minibatch[0][key] = [minibatch[0][key][i] for i in indices]

                    temp_minibatch[key] = minibatch[0][key]

            elif type(minibatch[0][key]) == dict:
                for inside_key in minibatch[0][key].keys():
                    minibatch[0][key][inside_key] = \
                        torch.cat([minibatch[0][key][inside_key], minibatch[1][key][inside_key]], dim=0)[indices]

                    temp_minibatch[key] = minibatch[0][key]

            else:
                raise Exception('Could not find the type for this data')

        return temp_minibatch


def fix_output_dir_due_different_user(model_user_name: str, output_dir):
    if sys.platform != 'linux':
        return output_dir

    path_parts = os.getcwd().split('/')
    if 'womer' in path_parts:
        current_user = 'Omer'

    elif 'rschley' in path_parts:
        current_user = 'Ran'

    elif 'sglis' in path_parts:
        current_user = 'Gil'

    else:
        raise Exception('current user is not recognized')

    if model_user_name != current_user:
        if current_user == 'Omer':
            output_dir = os.path.join(r'/home/womer/project/runs', 'from_other_user', output_dir.split('/')[-1])

        elif current_user == 'Ran':
            output_dir = os.path.join(r'/home/rschley/code/WSI_MIL/WSI_MIL/runs', 'from_other_user',
                                      output_dir.split('/')[-1])

        elif current_user == 'Gil':
            output_dir = os.path.join(r'/mnt/gipnetapp_public/sgils/ran/runs', 'from_other_user',
                                      output_dir.split('/')[-1])

        # Create the Path to which data will be saved:

        Path(output_dir).mkdir(parents=True)

    return output_dir