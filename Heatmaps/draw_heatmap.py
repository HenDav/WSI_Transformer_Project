import pandas as pd
import matplotlib.pyplot as plt
import openslide
from PIL import Image
import numpy as np
import cv2
import os

dn = r'C:\ran_data\TCGA_lung'
n_file = 1

if n_file == 1:
    sf = 'TCGA-55-7726-11A-01-TS1.e55bcf9f-bc93-4a0c-bdae-e770414bb32e.svs'
elif n_file == 2:
    sf = 'TCGA-05-5420-11A-01-TS1.062c76b9-163d-4a4a-963d-ca2d56bddaa7.svs'
elif n_file == 3:
    sf = 'TCGA-64-1679-01A-01-BS1.06dc184d-224d-4602-8194-fcc9392033f4.svs'
f = 'is_cancer_BatchOfSlides_Exp_375_Epoch_400_Inference_Full_Slide_' + sf + '_ScoreHeatMap.xlsx'
slide_file = os.path.join(dn, 'TCGA_LUNG', sf)
file = os.path.join(dn, '', f)

heatmap_DF = pd.read_excel(file)
heatmap_DF.drop('Unnamed: 0', inplace=True, axis=1)
heatmap = heatmap_DF.to_numpy()
heatmap[heatmap==-1] = np.nan
heatmap *= 255

#binary_heatmap
bin_heatmap = heatmap.copy()
bin_heatmap[bin_heatmap > 0.8] = 1
bin_heatmap[bin_heatmap < 0.2] = 0
bin_heatmap[(bin_heatmap > 0.2) & (bin_heatmap < 0.8)] = np.nan


#heatmap_im = Image.fromarray(heatmap.astype(int),'L')

slide = openslide.OpenSlide(slide_file)
objective_pwr = int(slide.properties['aperio.AppMag'])
magnification = 5
tile_size = 256
height = slide.dimensions[1]
width = slide.dimensions[0]
width_thumb = int(width / (objective_pwr / magnification))
height_thumb = int(height / (objective_pwr / magnification))
thumb = slide.get_thumbnail((width_thumb, height_thumb))

heatmap_resized = cv2.resize(heatmap, dsize=(width_thumb, height_thumb), interpolation=cv2.INTER_NEAREST)

fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
ax1.imshow(thumb)
ax2.imshow(heatmap_resized, cmap='jet')
#ax2.imshow(heatmap, cmap='jet')
plt.show()

print()



