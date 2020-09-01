# this model received assistance from the Tiramisu model
# https://github.com/0bserver07/One-Hundred-Layers-Tiramisu

from  __future__ import absolute_import
from __future__ import print_function

import numpy as np
import json

import tensorflow as tf
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer,Dense,Dropout,\
    Activation,Flatten,Reshape,Permute
from tensorflow.keras.layers import Conv2D,MaxPooling2D,UpSampling2D,Cropping2D
from keras.layers.normalization import BatchNormalization

from tensorflow.keras.layers import Conv2D,Conv2DTranspose

from keras import backend as kb

import cv2

# weight decay = 0.0001
l2 = tf.keras.regularizers.l2(0.0001)

class Tiramisu :
    """
    this will build the core model that we will use in our software
    """

    # initialize our model
    def __init__(self):
        self.model = Sequential()

    # now create our layer-creation functions
    def DenseBlock(self,layers,filters):
        model = self.model
        for i in range(layers) :
            model.add(BatchNormalization(
                model=0,axis=1,
                gamma_regularizer=l2,
                beta_regularizer=l2
            ))
            model.add(Activation('relu'))
            model.add(Conv2D(filters,kernel_size=(3,3),padding='same',
                             kernel_initializer='he_uniform',
                             data_format='channels_last'))
            model.add(Dropout(.2))

    def TransitionDown(self,filters):
        model = self.model
        model.add(BatchNormalization(mode=0,axis=1,
                                     gamma_regularizer=l2,
                                     beta_regularizer=l2))
        model.add(Activation('relu'))
        model.add(Conv2D(filters,kernel_size=(1,1),padding='same',
                         kernel_initializer='he_uniform'))
        model.add(Dropout(.2))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),
                               data_format='channels_last'))

    def TransitionUp(self,filters,input_shape,output_shape):
        model = self.model
        model.add(Conv2DTranspose(filters,kernel_size=(3,3),strides=(2,2),
                                  padding='same',
                                  input_shape=input_shape,
                                  output_shape=output_shape,
                                  kernel_initializer='he_uniform',
                                  data_format='channels_last'))

    def create(self):
        model = self.model
        model.add(Conv2D(48, kernel_size=(3,3),
                   padding='same',
                   input_shape=(224,224,3),
                   kernel_initializer='he_uniform',
                   kernel_regularizer=l2,
                   data_format='channels_last'
            ))
