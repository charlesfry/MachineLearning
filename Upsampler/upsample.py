import numpy as np
import pandas as pd
import tensorflow as tf
import os
from PIL import Image
from pathlib import Path
from tensorflow.keras import layers

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

TRAIN_DIR = './input/scale_data/train/train_scale'
TEST_DIR = './input/scale_data/labels/labels_scale'

train_img_width,train_img_height = get_image_sizes(TRAIN_DIR)
test_img_width,test_img_height = get_image_sizes(TEST_DIR)

batch_size = 32

train = tf.data.Dataset.list_files(F'{TRAIN_DIR}/*.jpg')
labels = tf.data.Dataset.list_files(F'{TEST_DIR}/*.jpg')

input_shape = (320, 240)
model = tf.keras.models.Sequential([
    # layers.Input(batch_size=batch_size),
    layers.experimental.preprocessing.Rescaling(1/255),
    layers.Conv2D(32,3,activation='relu'),
    layers.Dense(640*480,activation='linear'),
    layers.experimental.preprocessing.Rescaling(255)
])

model.compile(
    optimizer='adam',
    loss='mse'
)

model.fit(train,labels,epochs=50)
model.evaluate(train,labels)
