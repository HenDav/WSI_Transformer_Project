from Dataset_Maker.dataset_utils import name_of_gipdeep_host_node
import utils
from utils import get_patches_with_overlap
import sys
import argparse
from Nets import nets, PreActResNets, resnet_v2
import torch
import os
import numpy as np
from tqdm import tqdm
import random
from pathlib import Path
import imageio
import openslide
from PIL import Image

random.seed(0)
parser = argparse.ArgumentParser(description='WSI_MIL Slide inference with visualization for Regular model')
parser.add_argument('-ex', '--experiment', type=int, default=40015, help='model to use')
parser.add_argument('-fe', '--from_epoch', type=int, default=1000, help='model to use')
parser.add_argument('-sn', '--slide_name', type=str, default='17-10005_1_1_a', help='slide to use')
parser.add_argument('-dp', '--dataset_path', type=str, default='/Breast/Carmel/1-8/Batch_1/CARMEL1/', help='path to dataset folder')
args = parser.parse_args()

gipdeep10_used = name_of_gipdeep_host_node() == "gipdeep10"
if gipdeep10_used:
    path_init = r'/data'
else:
    path_init = r'/mnt/gipmed_new/Data'

slide_file = os.path.join(path_init+args.dataset_path, args.slide_name+'.mrxs')
segMap_file = os.path.join(path_init+args.dataset_path, 'SegData/SegMaps/', args.slide_name + '_SegMap.png')

DEVICE = utils.device_gpu_cpu()
data_path = ''
downsample_rate = 8
frame_size = int(64 / downsample_rate)


# Load saved model:
print('Loading pre-saved model from Exp. {} and Epoch {}'.format(args.experiment, args.from_epoch))
run_data_output = utils.run_data(experiment=args.experiment)
output_dir, TILE_SIZE, args.target, model_name, desired_magnification =\
    run_data_output['Location'], run_data_output['Tile Size'], run_data_output['Receptor'],\
    run_data_output['Model Name'], run_data_output['Desired Slide Magnification']

model = eval(model_name)

# loading model parameters from the specific epoch
model_data_loaded = torch.load(os.path.join(output_dir, 'Model_CheckPoints',
                                            'model_data_Epoch_' + str(args.from_epoch) + '.pt'), map_location='cpu')
model.load_state_dict(model_data_loaded['model_state_dict'])
model.eval()

model.is_HeatMap = True


new_slide = True


# Create folders to save the data:

#path_for_output = 'Inference/Full_Slide_Inference/' + inf_dset.DataSet
path_for_output = os.path.join('./Visualizations', 'exp_'+str(args.experiment), 'Epoch_'+str(args.from_epoch),args.slide_name+ '.mrxs')
print('path is:' + path_for_output)
Path(path_for_output).mkdir(parents=True, exist_ok=True)


with torch.no_grad():
    for batch_idx, (data, window_top_left_in_level, window_size, equivalent_grid_size) in enumerate(tqdm(get_patches_with_overlap(slide_file, segMap_file, rate = downsample_rate, size = 2048))):

        if new_slide:
            print('Working on Slide {}'.format(args.slide_name))
            equivalent_slide_heat_map = np.ones((equivalent_grid_size)) * (-1)  # This heat map should be filled with the weights.
            new_slide = False
        data = data.to(DEVICE)
        # target = target.to(DEVICE)

        model.to(DEVICE)
        #output = (1/(1+torch.exp(-model(data).squeeze()))).cpu().numpy()
        output = torch.sigmoid(model(data)).squeeze().cpu().numpy()

        
        taken_window_top_left = (window_top_left_in_level[0] + frame_size, window_top_left_in_level[1] + frame_size)
        taken_window = output[frame_size:-frame_size, frame_size:-frame_size]
        taken_window_size = taken_window.shape
        
        #taken_window_size = (window_size[0] - 2*frame_size, window_size[1] - 2*frame_size)
        #print(taken_window_top_left[0])
        #print(taken_window_top_left[0] + taken_window_size[0])
        #print(taken_window_top_left[1])
        #print(taken_window_top_left[1] + taken_window_size[1])
        #print(equivalent_slide_heat_map.shape)
        #print(taken_window.shape)
        #print(equivalent_slide_heat_map[taken_window_top_left[0]:taken_window_top_left[0] + taken_window_size[0], taken_window_top_left[1]:taken_window_top_left[1] + taken_window_size[1]].shape)
        try:
            equivalent_slide_heat_map[taken_window_top_left[0]:taken_window_top_left[0] + taken_window_size[0], taken_window_top_left[1]:taken_window_top_left[1] + taken_window_size[1]] = taken_window
        except:
            shape = equivalent_slide_heat_map[taken_window_top_left[0]:taken_window_top_left[0] + taken_window_size[0], taken_window_top_left[1]:taken_window_top_left[1] + taken_window_size[1]].shape
            taken_window = taken_window[:shape[0],:shape[1]]
            equivalent_slide_heat_map[taken_window_top_left[0]:taken_window_top_left[0] + taken_window_size[0], taken_window_top_left[1]:taken_window_top_left[1] + taken_window_size[1]] = taken_window
            



# Save the heat map to file

slide = openslide.OpenSlide(slide_file)
height = slide.dimensions[1]
width = slide.dimensions[0]
mag_thumb = 5
height_thumb = np.min((int(height / mag_thumb), 10000))
width_thumb = int(width / height * height_thumb)

segmap = Image.open(segMap_file).resize((width_thumb, height_thumb))
thumb = slide.get_thumbnail((int(slide.dimensions[0]/4), int(slide.dimensions[1]/4)))

#basic_file_name = os.path.join(path_for_output,
#                               '_'.join([args.target, 'BatchOfSlides', 'Exp', str(args.experiment),
#                                         'Epoch', str(args.from_epoch), 'Inference_Full_Slide']))

heatmap_file = os.path.join(path_for_output,'ScoreHeatMap.png')
background_map_file = os.path.join(path_for_output,'BackgroundMap.png')
seg_file = os.path.join(path_for_output,'SegMap.png')
thumb_file = os.path.join(path_for_output,'Thumb.jpg')

background_map = (equivalent_slide_heat_map == -1)
equivalent_slide_heat_map[equivalent_slide_heat_map == -1] = 0
imageio.imwrite(heatmap_file, (equivalent_slide_heat_map * 65535).astype(np.uint16))
imageio.imwrite(background_map_file, (background_map * 255).astype(np.uint8))
imageio.imwrite(seg_file, segmap)
imageio.imwrite(thumb_file, thumb)



print('Done !')