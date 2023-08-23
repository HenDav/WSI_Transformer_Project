import aicsimageio
import numpy as np
import pyvips

# Read the CZI file
img = aicsimageio.imread("your_file.czi")

# Convert to 8-bit (tifffile and pyvips require 8-bit or 16-bit)
img_8bit = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)

# Create a new pyvips image from the numpy array
vips_image = pyvips.Image.new_from_memory(img_8bit.tobytes(), img_8bit.shape[1], img_8bit.shape[0], bands=1, format='uchar')

# Save the image as a pyramidal TIFF
vips_image.tiffsave("output_file.tiff", tile=True, pyramid=True, compression='jpeg', Q=90)
