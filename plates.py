
# coding: utf-8

# # Mask R-CNN - Train on Shapes Dataset
#
#
# This notebook shows how to train Mask R-CNN on your own dataset. To keep things simple we use a synthetic dataset of shapes (squares, triangles, and circles) which enables fast training. You'd still need a GPU, though, because the network backbone is a Resnet101, which would be too slow to train on a CPU. On a GPU, you can start to get okay-ish results in a few minutes, and good results in less than an hour.
#
# The code of the *Shapes* dataset is included below. It generates images on the fly, so it doesn't require downloading any data. And it can generate images of any size, so we pick a small image size to train faster.

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

from config import Config
import utils
import model as modellib
import visualize
from model import log
import csv
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#get_ipython().run_line_magic('matplotlib', 'inline')

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


# ## Configurations

# In[2]:


class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "plates"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + ` shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 200

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 3000

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50

    LEARNING_RATE = 0.002
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization
    #WEIGHT_DECAY = 0.0002

    # Gradient norm clipping
    GRADIENT_CLIP_NORM = 6.0

config = ShapesConfig()
config.display()

# ## Notebook Preferences

# In[3]:


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


# ## Dataset
#
# Create a synthetic dataset
#
# Extend the Dataset class and add a method to load the shapes dataset, `load_shapes()`, and override the following methods:
#
# * load_image()
# * load_mask()
# * image_reference()

# In[4]:


class ShapesDataset(utils.Dataset):
    """Generates the plates synthetic dataset. The dataset consists of
    plates placed randomly on a random image.
    The images are already generated.
    """

    def load_shapes(self, val=False):
        """Load the requested number of synthetic images.
        """
        # Add classes
        self.add_class("plates", 1, "plate")

        # Add images '/home/mateusz/workspace/tablice_oznaczone.csv'
        with open('train/tablice_oznaczone.csv', newline='') as csvfile:
            CSVreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
            rows = list(CSVreader)
            if(val == True):
                rows = rows[-len(rows)//20:]
                print("Val: ", len(rows))
            else:
                rows = rows[:-len(rows)//20]
                print("Train: ", len(rows))

            for n, col in enumerate(rows):
                if(col[0] == ''):
                    continue
                file_path = col[0]
                pts = []
                for y in range(2, len(col)):
                    if col[y] == '':
                        continue
                    p = col[y].split(';')
                    pts.append(np.array([x.split(',') for x in p if not x == ''], dtype=np.int32))

                self.add_image('plates', n, 'train/' + file_path, pkt = pts)

    def load_image(self, image_id):
        """Load image from file """
        image = cv2.imread(self.image_info[image_id]['path'])
        return image

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "plates":
            return info["plates"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for plates of the given image ID.
        """

        image = cv2.imread(self.image_info[image_id]['path'])
        rows, cols = image.shape[:2]
        pkt = self.image_info[image_id]['pkt']
        liczba_masek = len(pkt)
        mask = np.zeros([rows, cols, liczba_masek], dtype=np.uint8)
        for y in range(liczba_masek):
            mask[:,:,y] = cv2.fillPoly(mask[:,:,y].copy(), pts = [pkt[y]], color=(255,255,255))

        return mask.astype(np.bool), np.full(liczba_masek, 1) #np.array([1, 1]) #np.arange(1, dtype = np.int32)


# In[5]:

# Training dataset
dataset_train = ShapesDataset()
dataset_train.load_shapes(val=False)
dataset_train.prepare()

# Validation dataset
dataset_val = ShapesDataset()
dataset_val.load_shapes(val=True)
dataset_val.prepare()

# In[6]:


# Load and display random samples
#image_ids = [0, 1, 2] #np.random.choice(dataset_train.image_ids, 4)
#image_ids = np.random.choice(dataset_train.image_ids, 2)
image_ids = []
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)
#sys.exit(0)
# ## Create Model

# In[ ]:


# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)


# In[7]:


# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)


# ## Training
#
# Train in two stages:
# 1. Only the heads. Here we're freezing all the backbone layers and training only the randomly initialized layers (i.e. the ones that we didn't use pre-trained weights from MS COCO). To train only the head layers, pass `layers='heads'` to the `train()` function.
#
# 2. Fine-tune all layers. For this simple example it's not necessary, but we're including it to show the process. Simply pass `layers="all` to train all layers.

# In[8]:


# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=120,
            layers='heads')


# In[9]:

# Fine tune all layers
# Passing layers="all" trains all layers. You can also
# pass a regular expression to select which layers to
# train by name pattern.
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=122,
            layers="all")

# In[10]:


# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
#model_path = os.path.join(MODEL_DIR, "mask_rcnn_plates.h5")
#model.keras_model.save_weights("mask_rcnn_plates.h5")
#print("Model zapisany: mask_rcnn_plates.h5")
print('Zakonczono pomyslnie')
