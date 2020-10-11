import numpy as np
import pandas as pd
import tensorflow as tf
import os
from PIL import Image
from pathlib import Path

def seed_everything(seed=0):
    #random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

seed = 69
seed_everything(seed)

def get_image_sizes(dir) :
    """
    this function returns the size of the first element of the directory and returns the size
    only necessary if we dont know the sizes of the images
    """
    for root, _, files in os.walk(dir):
        for file in files:
            path = os.path.join(root, file)
            try :
                img = Image.open(path)
                return img.size
            except :
                continue
    print(f"Error: no images found in directory {dir}")
    return None

TRAIN_DIR = './input/scale_data/train_scale'
TEST_DIR = './input/scale_data/test_scale'

train_img_width,train_img_height = get_image_sizes(TRAIN_DIR)
test_img_width,test_img_height = get_image_sizes(TEST_DIR)

batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    './input/scale_data/',
    validation_split=0.2,
    subset="training",
    seed=seed,
    image_size=(train_img_width,train_img_height),
    batch_size=batch_size
)
