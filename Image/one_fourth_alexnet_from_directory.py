# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 08:28:01 2018

@author: Chastine
"""

from keras.models import Sequential, load_model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import os
import cv2
from PIL import Image
from PIL.Image import core as _imaging
from quiver_engine import server

class LocalResponseNormalization(Layer):
    
    def __init__(self, n=5, alpha=0.0005, beta=0.75, k=2, **kwargs):
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.k = k
        super(LocalResponseNormalization, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.shape = input_shape
        super(LocalResponseNormalization, self).build(input_shape)
        
    def call(self, x, mask=None):
        if K.image_dim_ordering == "th":
            _, f, r, c = self.shape
        else:
            _, r, c, f = self.shape
        squared = K.square(x)
        pooled = K.pool2d(squared, (self.n, self.n), strides=(1, 1),
                          padding="same", pool_mode="avg")
        if K.image_dim_ordering == "th":
            summed = K.sum(pooled, axis=1, keepdims=True)
            averaged = self.alpha * K.repeat_elements(summed, f, axis=1)
        else:
            summed = K.sum(pooled, axis=3, keepdims=True)
            averaged = self.alpha * K.repeat_elements(summed, f, axis=3)
        denom = K.pow(self.k + averaged, self.beta)
        return x / denom
    
    def get_output_shape_for(self, input_shape):
        return input_shape


filepath = "weights/cnn/one_fourth_alexnet/weights@epoch-{epoch:03d}-{val_acc:.2f}.hdf5"

value_shift = 0.1
value_range = 0.2
rotation_degree = 20

tensorboard = TensorBoard(log_dir='./Graph/one_fourth_alexnet', histogram_freq=0,  
          write_graph=True, write_images=True)
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', verbose=1, patience=10)

callbacks = [tensorboard, checkpoint]

#callbacks = [tensorboard, checkpoint, early_stopping]

train_set = '../dataset/image/training_set/'

test_set = '../dataset/image/test_set/'

dirs = next(os.walk(train_set))[1]

classes = dirs
num_classes = len(dirs)

classifier = Sequential()

classifier.add(Conv2D(24, (11, 11), strides = 4, input_shape = (227, 227, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (3, 3), strides = 2))
classifier.add(LocalResponseNormalization())
classifier.add(Conv2D(64, (5, 5), strides = 1, padding = "same", activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (3, 3), strides = 2))
classifier.add(LocalResponseNormalization())
classifier.add(Conv2D(96, (3, 3), strides = 1, padding = "same", activation = 'relu'))
classifier.add(Conv2D(96, (3, 3), strides = 1, padding = "same", activation = 'relu'))
classifier.add(Conv2D(64, (3, 3), strides = 1, padding = "same", activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (3, 3), strides = 2))
classifier.add(Flatten())
classifier.add(Dropout(0.5))
classifier.add(Dense(units = 1024, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units = 1024, activation = 'relu'))
classifier.add(Dense(units = num_classes, activation = 'softmax'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = value_range,
                                   zoom_range = value_range,
                                   rotation_range = rotation_degree,
                                   width_shift_range = value_shift,
                                   height_shift_range = value_shift,
                                   horizontal_flip = True,
                                   samplewise_center = True,
                                   samplewise_std_normalization = True)

test_datagen = ImageDataGenerator(rescale = 1./255,
                                  shear_range = value_range,
                                  zoom_range = value_range,
                                  rotation_range = rotation_degree,
                                  width_shift_range = value_shift,
                                  height_shift_range = value_shift,
                                  horizontal_flip = True,
                                  samplewise_center = True,
                                  samplewise_std_normalization = True)

training_set = train_datagen.flow_from_directory(train_set,
                                                 target_size = (227, 227),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(test_set,
                                            target_size = (227, 227),
                                            batch_size = 32,
                                            class_mode = 'categorical')

#training_set = train_datagen.flow(x_train, y_train)
#test_set = test_datagen.flow(x_test, y_test)

#training_set = (x_train, y_train)
#test_set = (x_test, y_test)

#classifier.fit_generator(training_set,
#                         steps_per_epoch = 3000,
#                         epochs = 1000,
#                         validation_data = test_set,
#                         validation_steps = 600, callbacks = callbacks)

x_generated, y_generated = training_set.next()

for x in x_gen:
    image_to_save = 

#classifier.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 50, callbacks = callbacks)


