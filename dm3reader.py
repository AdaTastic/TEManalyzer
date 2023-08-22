import ncempy.io as nio
import numpy as np

def getDM3Image(file_path):
    data = nio.read(file_path)
    image_array = data['data']
    image_array = image_array/100
    image_array = np.clip(image_array, 0, 255).astype(np.uint8)
    array_height, array_width = image_array.shape
    rgba_array = np.empty((array_height, array_width, 4), dtype=np.uint8)
    rgba_array[:, :, 0] = image_array  # Copy grayscale values to the red channel
    rgba_array[:, :, 1] = image_array  # Copy grayscale values to the green channel
    rgba_array[:, :, 2] = image_array  # Copy grayscale values to the blue channel
    rgba_array[:, :, 3] = 255    # Set alpha to 255 (fully opaque)
    return rgba_array

def getDM3scale(file_path):
    data = nio.read(file_path)
    pixelUnit = data['pixelUnit'][0]
    pixelSize = float(data['pixelSize'][0])
    return pixelUnit, pixelSize

