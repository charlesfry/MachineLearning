# this model received assistance from the original Tiramisu model
# https://github.com/0bserver07/One-Hundred-Layers-Tiramisu

from  __future__ import absolute_import
from __future__ import print_function

import numpy as np
import pickle

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
        self.create()

    # now create our layer-creation functions
    def DenseBlock(self,layers,filters):
        for i in range(layers) :
            self.model.add(BatchNormalization(
                axis=1,
                gamma_regularizer=l2,
                beta_regularizer=l2
            ))
            self.model.add(Activation('relu'))
            self.model.add(Conv2D(filters, kernel_size=(3, 3), padding='same',
                                  kernel_initializer='he_uniform',
                                  data_format='channels_last'))
            self.model.add(Dropout(.2))

    def TransitionDown(self,filters):
        self.model.add(BatchNormalization(axis=1,
                                          gamma_regularizer=l2,
                                          beta_regularizer=l2))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(filters, kernel_size=(1, 1), padding='same',
                              kernel_initializer='he_uniform'))
        self.model.add(Dropout(.2))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                                    data_format='channels_last'))

    def TransitionUp(self,filters,input_shape,output_shape):
        self.model.add(Conv2DTranspose(filters, kernel_size=(3, 3), strides=(2, 2),
                                       padding='same',
                                       input_shape=input_shape,
                                       #output_shape=output_shape,
                                       kernel_initializer='he_uniform',
                                       data_format='channels_last'))

    def create(self):
        self.model = Sequential()
        self.model.add(Conv2D(48, kernel_size=(3, 3),
                              padding='same',
                              input_shape=(224,224,3),
                              kernel_initializer='he_uniform',
                              kernel_regularizer=l2,
                              data_format='channels_last')
                       )
        # (5 * 4)* 2 + 5 + 5 + 1 + 1 +1
        # growth_m = 4 * 12
        # previous_m = 48
        self.DenseBlock(5, 108)  # 5*12 = 60 + 48 = 108
        self.TransitionDown(108)
        self.DenseBlock(5, 168)  # 5*12 = 60 + 108 = 168
        self.TransitionDown(168)
        self.DenseBlock(5, 228)  # 5*12 = 60 + 168 = 228
        self.TransitionDown(228)
        self.DenseBlock(5, 288)  # 5*12 = 60 + 228 = 288
        self.TransitionDown(288)
        self.DenseBlock(5, 348)  # 5*12 = 60 + 288 = 348
        self.TransitionDown(348)

        self.DenseBlock(15, 408)  # m = 348 + 5*12 = 408

        self.TransitionUp(468, (468, 7, 7), (None, 468, 14, 14))  # m = 348 + 5x12 + 5x12 = 468.
        self.DenseBlock(5, 468)

        self.TransitionUp(408, (408, 14, 14), (None, 408, 28, 28))  # m = 288 + 5x12 + 5x12 = 408
        self.DenseBlock(5, 408)

        self.TransitionUp(348, (348, 28, 28), (None, 348, 56, 56))  # m = 228 + 5x12 + 5x12 = 348
        self.DenseBlock(5, 348)

        self.TransitionUp(288, (288, 56, 56), (None, 288, 112, 112))  # m = 168 + 5x12 + 5x12 = 288
        self.DenseBlock(5, 288)

        self.TransitionUp(228, (228, 112, 112), (None, 228, 224, 224))  # m = 108 + 5x12 + 5x12 = 228
        self.DenseBlock(5, 228)

        self.model.add(Conv2D(12, kernel_size=(1, 1),
                              padding='same',
                              kernel_initializer="he_uniform",
                              kernel_regularizer=l2,
                              data_format='channels_last'))

        self.model.add(Reshape((12, 224 * 224)))
        self.model.add(Permute((2, 1)))
        self.model.add(Activation('softmax'))
        self.model.summary()

        self.model.save('tiramisu')

        return self.model

Tiramisu()