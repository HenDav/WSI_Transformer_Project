import openslide
import matplotlib.pyplot as plt
import numpy as np
import time
import sys

if sys.platform == 'win32':
    file_ndpi = r'C:\ran_data\ABCTB\ABCTB_examples\ABCTB\01-06-034.001.EX.B1.ndpi'
    file_tif = r'C:\ran_data\ABCTB\ABCTB_examples\ABCTB\01-06-034.001.EX.B1.tif'
    file_mrxs = r'C:\ran_data\ABCTB\ABCTB_examples\ABCTB\01-06-034.001.EX.B1.mrxs'
else:
    '''file_ndpi = r'/mnt/hdd/slide_compare/01-06-034.001.EX.B1.ndpi'
    file_tif = r'/mnt/hdd/slide_compare/01-06-034.001.EX.B1.tif'
    file_mrxs = r'/mnt/hdd/slide_compare/01-06-034.001.EX.B1.mrxs' '''
    file_ndpi = r'/home/rschley/temp/slide_compare/01-06-034.001.EX.B1.ndpi'
    file_tif = r'/home/rschley/temp/slide_compare/01-06-034.001.EX.B1.tif'
    file_mrxs = r'/home/rschley/temp/slide_compare/01-06-034.001.EX.B1.mrxs'

slide_ndpi = openslide.OpenSlide(file_ndpi)
slide_tif = openslide.OpenSlide(file_tif)
slide_mrxs = openslide.OpenSlide(file_mrxs)

time1 = time.time()
tile_ndpi = (slide_ndpi.read_region(location=(18000, 11500), level=0, size=(512, 512))).convert('RGB')
time2 = time.time()
tile_tif = (slide_tif.read_region(location=(18000, 11500), level=0, size=(512, 512))).convert('RGB')
time3 = time.time()
tile_mrxs = (slide_mrxs.read_region(location=(18000, 11500), level=0, size=(512, 512))).convert('RGB')
time4 = time.time()

print('time to load ndpi patch: ', str(time2-time1), ' sec')
print('time to load tif patch: ', str(time3-time2), ' sec')
print('time to load mrxs patch: ', str(time4-time3), ' sec')

fig,ax = plt.subplots(1,3)
ax[0].imshow(tile_ndpi)
ax[1].imshow(tile_tif)
ax[2].imshow(tile_mrxs)
ax[0].set_title('ndpi')
ax[1].set_title('tiff')
ax[2].set_title('mrxs')

diff1 = np.sum(np.abs(np.array(tile_ndpi) - np.array(tile_tif)))
diff2 = np.sum(np.abs(np.array(tile_ndpi) - np.array(tile_mrxs)))

print('diff1 = ', str(diff1))
print('diff2 = ', str(diff2))

ndpi_arr = np.array(tile_ndpi, dtype=int)
mrxs_arr = np.array(tile_mrxs, dtype=int)
diff22 = np.sum(np.abs(ndpi_arr - mrxs_arr))

fig,ax = plt.subplots(1, 3, sharex=True, sharey=True)
ax[0].imshow(tile_ndpi)
ax[1].imshow((ndpi_arr - mrxs_arr)*20)
ax[2].imshow(tile_mrxs)
ax[0].set_title('ndpi')
ax[1].set_title('diff2')
ax[2].set_title('mrxs')

print('finished')