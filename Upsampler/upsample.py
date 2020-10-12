import numpy as np
import pandas as pd
import tensorflow as tf
import os
from PIL import Image
from pathlib import Path
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
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

TRAIN_DIR = './input/scale_data/train/butt'
LABEL_DIR = './input/scale_data/labels/butt'

train_img_width,train_img_height = get_image_sizes(TRAIN_DIR)
train_size = (train_img_height,train_img_width)
test_img_width,test_img_height = get_image_sizes(LABEL_DIR)
label_size = (test_img_height,test_img_width)



# train = tf.data.Dataset.list_files(f'{TRAIN_DIR}/*.jpg')
# labels = tf.data.Dataset.list_files(f'{LABEL_DIR}/*.jpg')


class DataGenerator(Sequence):

    def __init__(self, base_dir, label_dir, output_size, label_size, shuffle=False, batch_size=1):
        """
        Initializes a data generator object
        :param base_dir: the directory in which all images are stored
        :param label_dir: the directory in which the target images are stored
        :param output_size: image output size after preprocessing
        :param shuffle: shuffle the data after each epoch
        :param batch_size: The size of each batch returned by __getitem__
        """
        self.base_dir = base_dir
        self.label_dir = label_dir
        self.output_size = output_size
        self.label_size = label_size
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.df = pd.DataFrame(data=os.listdir(base_dir))

        if label_dir is None : self.label_dir = base_dir
        self.on_epoch_end()



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
            if img.size != self.output_size[::-1] :
                img = img.resize(self.output_size[::-1])

            assert label.size == self.label_size[::-1], f'Assertion failed: label size must be {(self.label_size[1],self.label_size[0])}. ' \
                                                  f'Label {label_path} has size {label.size}'

            ## if you have any preprocessing for
            ## the labels too do it here
            label = np.array(label) / 255

            X[i,] = img
            y[i] = label

        return X, y


batch_size = 1
ds_train = DataGenerator(base_dir=TRAIN_DIR,label_dir=LABEL_DIR,output_size=train_size,label_size=label_size,
                         shuffle=False,batch_size=batch_size)

# build the model
model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(train_img_height, train_img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  # layers.Dense(label_size[0]*label_size[1]*3)
])

model.compile(
    optimizer='adam',
    loss='mse'
)

model.fit(ds_train,epochs=1,steps_per_epoch=1)
