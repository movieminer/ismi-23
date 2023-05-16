import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

from torch import nn
import os
from pathlib import Path
import SimpleITK as sitk
from medpy.io import load

# Some constants
data_dir = Path("./luna23-ismi-datasets/")
noduleTypes = ["non-solid", "part-solid", "solid", "calcified"]
n_classes = len(noduleTypes)


# Get all files in a directory, returning the file paths and names
def get_file_list(path,ext='',queue=''):
    if ext != '': return [os.path.join(path,f) for f in os.listdir(path) if f.endswith(''+queue+'.'+ext+'')],  [f for f in os.listdir(path) if f.endswith(''+queue+'.'+ext+'')]    
    else: return [os.path.join(path,f) for f in os.listdir(path)]


# Get all file paths and names of the training and test sets
train_dir = data_dir / "train_set"
train = get_file_list(train_dir / "images", ext='mha')
train_pixellabels = get_file_list(train_dir / "labels", ext='mha')
train_labels = get_file_list(train_dir, ext='csv')

test_dir = data_dir / "test_set"
test = get_file_list(test_dir / "images", ext='mha')


# Function to obtain orthogonal patches from a 3D image
def get_orthogonal_patches(x):
    dims = x.shape
    axial = x[dims[0]//2,:,:]. squeeze()
    coronal = x[:,:,dims[2]//2].squeeze()
    sagittal= x[:,dims[1]//2,:].squeeze()
    return axial, coronal, sagittal


# Function to plot 3D images in 2D with a slider to change the layer
def slider(x):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    axlayer = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider_layer = Slider(axlayer, 'Layer', 0, x.shape[2]-1, valinit=0, valstep=1)

    def update(val):
        layer = slider_layer.val
        ax.imshow(x[:,:,layer])
        fig.canvas.draw_idle()

    slider_layer.on_changed(update)

    ax.imshow(x[:,:,0])

    plt.show()


# for idx in range(len(mhas[0])):
for idx in range(0, 1):
    file_path = train[0][idx]
    file_name = train[1][idx]

    mha_data, mha_header = load(file_path)
