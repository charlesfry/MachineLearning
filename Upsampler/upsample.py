import numpy as np
import pandas as pd
import tensorflow as tf
import os
from PIL import Image
from pathlib import Path
from tensorflow.keras import layers
from tensorflow.keras.utils import Sequence
import matplotlib.image as mpimg

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
LABEL_DIR = './input/scale_data/labels/labels_scale'

train_img_width,train_img_height = get_image_sizes(TRAIN_DIR)
test_img_width,test_img_height = get_image_sizes(LABEL_DIR)

batch_size = 32

# train = tf.data.Dataset.list_files(f'{TRAIN_DIR}/*.jpg')
# labels = tf.data.Dataset.list_files(f'{LABEL_DIR}/*.jpg')

class DataGenerator(Sequence):

    def __init__(self, csv_file, base_dir,label_dir, output_size, label_size, shuffle=False, batch_size=batch_size):
        """
        Initializes a data generator object
        :param csv_file: file in which image names and numeric labels are stored
        :param base_dir: the directory in which all images are stored
        :param label_dir: the directory in which the target images are stored
        :param output_size: image output size after preprocessing
        :param shuffle: shuffle the data after each epoch
        :param batch_size: The size of each batch returned by __getitem__
        """
        self.df = pd.read_csv(csv_file)
        self.base_dir = base_dir
        self.label_dir = label_dir
        self.output_size = output_size
        self.label_size = label_size
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.on_epoch_end()

        if label_dir is None : self.label_dir = base_dir


    def on_epoch_end(self):
        self.indices = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(len(self.df) / self.batch_size)

    def __getitem__(self, idx):
        ## Initializing Batch
        #  that one in the shape is just for a one channel images
        # if you use colored images set that to 3
        X = np.empty((self.batch_size, *self.output_size, 3))
        # (x, y, h, w)
        y = np.empty((self.batch_size, *self.label_size, 3))

        # get the indices of the requested batch
        indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        for i, data_index in enumerate(indices):
            img_path = os.path.join(self.base_dir,
                                    self.df.iloc[data_index, 0])
            label_path = os.path.join(self.label_dir,
                                      self.df.iloc[data_index,0])


            img = Image.open(img_path)
            label = Image.open(label_path)

            ## this is where you preprocess the image
            ## make sure to resize it to be self.output_size
            img = img.resize(self.output_size)

            assert label.size == self.label_size, f'Assertion failed: label size must be {self.label_size}. ' \
                                                  f'Label {label_path} has size {label.size}'

            ## if you have any preprocessing for
            ## the labels too do it here
            

            X[i,] = img
            y[i] = label

        return X, y


input_shape = (320, 240)

img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,rotation_range=20,validation_split=.2)
X_train = img_gen.flow_from_directory(f'./input/scale_data/train',batch_size=batch_size,
                                           classes=None,subset='training')
y_train = img_gen.flow_from_directory(f'./input/scale_data/labels',batch_size=batch_size,
                                           classes=None,subset='training')

image_count = len(os.listdir(f'{TRAIN_DIR}'))






model = tf.keras.models.Sequential([
    layers.Conv1D(16,3,activation='relu',kernel_initializer='HeNormal'),
    layers.UpSampling1D(size=2),
    layers.Dense(320*240*2,activation='relu')
])

model.compile(
    optimizer='adam',
    loss='mse'
)

model.fit(X_train,y_train,epochs=20,steps_per_epoch=batch_size)
