# will be used for detectron2 fasterrcnn model zoo name
from sahi.utils.detectron2 import Detectron2TestConstants

# import required functions, classes
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from PIL import Image, ImageDraw
import numpy as np
import os
import cv2
import json
import matplotlib.pyplot as plt
import math
import dm3reader
import h5py
import configreader

def read_config():
    compute_device = configreader.get_variable_from_ini("computation device")
    resizedSize = int(configreader.get_variable_from_ini("preview image size"))
    sliceSize = int(configreader.get_variable_from_ini("preview slice size"))
    return compute_device,resizedSize,sliceSize

def runPreviewPredict(image_path,model_path,config_path):
    compute_device,resizedSize,sliceSize = read_config()
    # define model
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='detectron2',
        model_path=model_path,
        config_path=config_path,
        confidence_threshold=0.5,
        image_size=resizedSize,
        device=compute_device, # 'cpu' or 'cuda:0'
    )

    # generate resized img
    if image_path.endswith(".dm3"):
        image_array = dm3reader.getDM3Image(image_path)
        image = Image.fromarray(image_array)
        image = image.convert("RGB")
    elif image_path.endswith(".hdf5"):
        # loading dm3 stuff
        dm3_dict = {}
        with h5py.File(image_path, "r") as f:
            group = f["my_group"]
            
            for key in group.keys():
                dm3_dict[key] = group[key][()]
        
        image_array = dm3_dict['data']
        image = Image.fromarray(image_array)
        image = image.convert("RGB")
    else:
        image = Image.open(image_path)

    image = image.resize((resizedSize,resizedSize),Image.LANCZOS)
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    dir_name = os.path.dirname(image_path)
    image_path = dir_name + "/preview/" + name + "_resized.png"
    image.save(image_path)

    # predict
    result = get_sliced_prediction(
        image_path,
        detection_model,
        slice_height = int(sliceSize),
        slice_width = int(sliceSize),
        overlap_height_ratio = 0.2,
        overlap_width_ratio = 0.2,
    )

    results = result.to_coco_annotations()
    return results, image_path

def filterParticles(image_path,results,original_path):
    compute_device,resizedSize,sliceSize = read_config()
    image = Image.open(image_path)
    h,w = image.size
    # Initialize the mask arrays
    mask_array = np.zeros((h, w, 2), dtype=np.uint8)
    filtered_particles_list = []  # Initialize an empty list to store filtered particles
    edge_cutoff = math.floor(w * 0.005) # remove particles cut off by edge

    # embed image size metadata
    if original_path.lower().endswith(".dm3"):
        pixelUnit,pixelSize = dm3reader.getDM3scale(original_path)
        height, width = dm3reader.getDM3Image(original_path).shape[:2]
    elif original_path.lower().endswith(".hdf5"):
        data={}
        with h5py.File(original_path, "r") as f:
            group = f["my_group"]
            for key in group.keys():
                data[key] = group[key][()]
        pixelSize = data['pixelSize']
        pixelUnit = data['pixelUnit'].decode('utf-8')
        height, width = data['data'].shape[:2]
    else:
        pixelSize = 1
        pixelUnit = "pixel"
        width, height = Image.open(original_path).size

    for index, particle in enumerate(results):
        bbox = particle['bbox']
        x, y, L, W = bbox

        # Check if the particle's bounding box is within 10 pixels of the image edges
        if (x <= edge_cutoff or y <= edge_cutoff or x + L >= w - edge_cutoff or y + W >= h - edge_cutoff):
            continue # particle is too close to the edge, might be cut off

        mask = np.zeros((h, w), dtype=np.uint8)
        coords = particle['segmentation'][0]

        # Reshape the coordinates array to 2D (x, y) pairs
        coords = np.array(coords)
        polygon_points = coords.reshape(-1, 2)

        # Convert the polygon points to a NumPy array with int32 data type (required by fillPoly)
        polygon_points = np.array([polygon_points], dtype=np.int32)

        # Fill the polygon in the mask with ones (True values)
        cv2.fillPoly(mask, polygon_points, 1)

        mask = mask > 0

        # if no overlap, just add it !
        if not np.any(mask_array[mask, 0] > 0):
            # Update the mask array
            mask_array[:, :, 0] += mask
            mask_array[:, :, 1] = np.where(mask, index+1, mask_array[:, :, 1])
            filtered_particles_list.append(index)
            continue
        
        # there is overlap :(
        mask = mask > 0
        overlap_coords = np.argwhere(mask_array[:, :, 0] > 0)
        this_conf = results[index]['score']
        x = overlap_coords[0][1]
        y = overlap_coords[0][0]
        other_index = mask_array[x,y,1] 
        other_conf = results[other_index]['score']

        # if this particle is worse, dont add it!
        if this_conf < other_conf:
            continue

        # Remove the old particle
        other_mask = mask_array[:, :, 1] == other_index + 1
        mask_array[other_mask, 0] = 0
        mask_array[other_mask, 1] = 0
        try:
            filtered_particles_list.remove(other_index)
        except ValueError:
            pass

        # add this particle
        mask_array[:, :, 0] += mask
        mask_array[:, :, 1] = np.where(mask, index+1, mask_array[:, :, 1])
        filtered_particles_list.append(index)

    # new filtered particle list
    filtered_particles = [{"pixelSize": pixelSize, "pixelUnit": pixelUnit,
                           "height": height, "width": width}]
    
    for i in filtered_particles_list:
        filtered_particles.append(results[i])

    # save to .txt
    preview_path = os.path.dirname(image_path)
    # preview_path = folder_path + "/preview/"
    image_filename = os.path.basename(image_path)
    name, ext = os.path.splitext(image_filename)
    output_file = preview_path + "/" + name + "_results.txt"
    with open(output_file, "w") as f:
        json.dump(filtered_particles, f)
    return output_file
    

def genPreview(image_path,results_path):
    compute_device,resizedSize,sliceSize = read_config()
    # Reading the data from the JSON file
    with open(results_path, "r") as f:
        results = json.load(f)

    image = Image.open(image_path)
    image = image.convert('RGBA')
    h,w = image.size
    mask_array = np.zeros((h, w, 4), dtype=np.uint8)

    # Create a draw object from the original image
    mask_image = Image.fromarray(mask_array, mode='RGBA')
    draw = ImageDraw.Draw(mask_image)

    class_list = ["", "cube", "hexagonal", "platelet", "platelet edge up"]
    color_mapping = {
        "cube": (255, 0, 0, 70),
        "hexagonal": (3, 248, 252, 70),
        "platelet": (252, 3, 86, 70),
        "platelet edge up": (255, 255, 0, 70)
    }

    results.pop(0)
    for particle in results:
        coords = particle['segmentation'][0]  # Assuming a single polygon per particle
        particle_class = class_list[particle['category_id']]
        if particle_class in color_mapping:
            color = color_mapping[particle_class]
            draw.polygon(coords, fill=color)


    # Create a new image by overlaying the mask on top of the original image
    overlay_image = Image.alpha_composite(image, mask_image)
    overlay_image = overlay_image.convert('RGB')

    preview_path = os.path.dirname(image_path)
    # preview_path = folder_path + "/preview/"
    image_filename = os.path.basename(image_path)
    name, ext = os.path.splitext(image_filename)
    output_file = preview_path + "/" + name + "_preview.jpg"
    overlay_image.save(output_file)

def genHistogram(txt_file_path):
    compute_device,resizedSize,sliceSize = read_config()
    with open(txt_file_path, "r") as f:
        results = json.load(f)

    rectangular_dims = []
    circular_dims =[]
    class_list = ["", "cube", "hexagonal", "platelet", "platelet edge up"]

    # get size info
    scaleInfo = results.pop(0)
    pixelUnit = scaleInfo['pixelUnit']
    pixelSize = scaleInfo['pixelSize']
    height = scaleInfo['height']
    width = scaleInfo['width']

    #correct size info
    for index, particle in enumerate(results):
        mask = np.array(particle['segmentation'][0])
        coords = mask.reshape(-1, 2)
        points = coords
        particle_class = class_list[particle['category_id']]

        # Find the minimum area rectangle that fits around the points
        if particle_class == "hexagonal":
            # Fit an ellipse to the polygon vertices
            if len(points) < 5:
                continue
            ellipse = cv2.fitEllipse(points)
            center, (major_axis, minor_axis), angle = ellipse
            minor_axis = height/resizedSize*minor_axis*pixelSize
            major_axis = height/resizedSize*major_axis*pixelSize
            circular_dims.append([index, minor_axis, major_axis])
            continue

        rect = cv2.minAreaRect(points)
        major_lentgh, minor_length = rect[1]
        major_lentgh = height/resizedSize*major_lentgh*pixelSize
        minor_length = height/resizedSize*minor_length*pixelSize
        rectangular_dims.append([index, major_lentgh, minor_length])

    # plot rectangulars
    # Extract the heights and widths from rectangular_dims
    heights = [item[1] for item in rectangular_dims]
    widths = [item[2] for item in rectangular_dims]

    # Combine the heights and widths into a single list representing lengths
    lengths = heights + widths

    # Plot the histogram
    plt.figure(1)
    plt.hist(lengths, bins=30, alpha=0.7, color='blue', label='Lengths')
    plt.xlabel(f"Length and Width ({pixelUnit})")
    plt.ylabel('Frequency')
    plt.title('Rectangulars')
    plt.legend()
    plt.grid(True)
    plt.show()

    # plot Circulars
    minor_r = [item[1] for item in circular_dims]
    major_r = [item[2] for item in circular_dims]

    # Combine the heights and widths into a single list representing lengths
    lengths = minor_r + major_r

    # Plot the histogram
    plt.figure(2)
    plt.hist(lengths, bins=30, alpha=0.7, color='blue', label='Lengths')
    plt.xlabel(f"Minor and Major Radius ({pixelUnit})")
    plt.ylabel('Frequency')
    plt.title('Circulars')
    plt.legend()
    plt.grid(True)
    plt.show()
