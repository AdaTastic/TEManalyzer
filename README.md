# TEManalyzer
## Table of Contents

- [Description](#description)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)

# Description

TEManalyzer is a GUI based AI image recognition model used to detect and size electron microscope images of perovskite nanoparticles. It can accept .jpg, .png, and .dm3 files as input. 

# Dependencies
The program dependencies and their recommended versions can be found in dependencies.txt. Aside from the python and torch/torchvision version, all packages should work with its latest versions.

The model can also be trained using the "modeltrainer.ipynb", if you would like to use it, a Jupyter notebook running python 3.10.8 is required. 

# Installation
download latest release and unzip
### For anaconda/miniconda
- install anaconda/miniconda
- add /conda folder to path
- add /conda/Scripts folder to path
- click "install_for_conda.bat" to automatically create virtual environment and install required packages. 

*Note: if installing detectron2 fails, it is likely because you need visual C++ 2014 and microsoft build tools. 

### For regular python
- install python 3.10.8
- install torch (v2.0.1) recommended
- install torchvision (v0.15.2) recommended
- install the rest of the packages in "dependencies.txt", any version should be fine

# Usage
If using conda, clicking "run_me_conda.bat" will start the created venv and open TEMGUI.py. Closing the GUI window will stop the venv. If not using conda, ensure dependencies are installed and run TEMGUI.py

## Opening images
Clicking open folder will allow you to select any folder containing your images. The image folder can be filled with .png, .jpg, or .dm3's and can be anywhere on the computer. Clicking the preview and output options will then generate preview and output folders inside the selected folder to output all the data.

## Editing images
This is particularly useful for uneditted dm3 files as the contrast and brightness usually needs to be adjusted. The opened images can then be scrolled through with "prev" and "next", each image can be adjusted using the two sliders. The left slider is contrast and the right one is brightness. If the image also has problem areas that you would like omitted from the detection, you can paint directly onto the image with a white brush with the mouse. The brush size can be adjusted with the "△" and "▽" buttons. 

  After editing, the result can be saved using the "save edits" button. Using this will overwrite the image in the folder if using .png or .jpg. If using .dm3, it will create a new .hdf5 file in the directory. 

## Preview
Selecting preview or analyze all images will create a folder within the image folder to output the data. Preview executes a low resolution detection of the image that takes roughly 30-60s on average to detect particles. Previewing is a good way to determine any problem areas with detection and paint them out as well as getting very rough estimates on the size distribution. 

Previewing results in outputs with lower accuracy in particle type detection due to the resolution descrepency between the low res image and the model. 

## Outputs
Selecting "analyzer all images(full)" will run the full detection on every image in the folder. If an image was edited and had its edits saved, that version will be used when running the detection. The results are then shown in /outputs and will include a image output, the txt results, and the sizing results. 

## Analyzing output dimensions
This example code shows how a histogram can be generated from the dimension.json file
```python
import matplotlib.pyplot as plt
import json

# open rect_dims.json or circ_dims.json
with open("dimensions.json", "r") as json_file:
    dimensions = json.load(json_file)
major_length = [item['major length'] for item in dimensions]
minor_length = [item['minor length'] for item in dimensions]

# Combine the heights and widths into a single list representing lengths
lengths = minor_length + major_length

# open filename_results.txt for length units
with open("results.txt", "r") as f:
        results = json.load(f)

# get size info
scaleInfo = results.pop(0)
lenUnit = scaleInfo['pixelUnit']

# plot
plt.figure()
plt.hist(lengths, bins=30, alpha=0.7, color='blue', label='Lengths')
plt.xlabel(f"Length and Width ({lenUnit})")
plt.ylabel('Frequency')
plt.title('Rectangulars')
plt.legend()
plt.grid(True)
plt.show()
```
Accessing data
```python
with open("dimensions.json", "r") as json_file:
    dimensions = json.load(json_file)
for particle in dimensions:
    particle_index = dimensions['index']
    major_length = dimensions['major length']
    minor_length = dimensions['minor length']
    particle_class = dimensions['particle class']
    particle_confidence = dimensions['conf']
```

## Switching models
The default model can be found in the model folder. A model is defined by the "model_final.pth" and "config.yaml" file. Switching models can be done by selecting the "select model folder" button and selecting a folder with these two files in them. 

## Training a custom model
Going into model/trainer/modeltrainer.ipynb opens up a Jupyter notebook that can train a custom model. The model training results are saved as a tensorboard in the output folder. The validation loss can then be compared to determine the optimal number of iterations to train the dataset on. Checkpointing is not availible yet.

## Editing classes
The class_list in "class_info.json" should match the classes in the dataset used to train the model. The class list can be found in the dataset as annotations_coco.json If additional classes are added or you wish to tweak the colors of each class in the output, edit "class_info.json" accordingly.

# License
This project is licensed under the [GNU General Public License v3.0](LICENSE).

