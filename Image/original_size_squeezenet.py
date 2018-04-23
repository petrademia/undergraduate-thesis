# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 12:58:38 2018

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
from quiver_engine import server
import SqueezeNet

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
early_stopping = EarlyStopping(monitor='val_loss', verbose=1, patience=5)

callbacks_without_early_stopping = [tensorboard, checkpoint]

callbacks = [tensorboard, checkpoint, early_stopping]

train_set = '../dataset/image/training_set/'

test_set = '../dataset/image/test_set/'

file_to_write = open('log.txt', mode='w+')

dirs = next(os.walk(train_set))[1]
num_classes = len(dirs)


total_train_set = 0
current_index = 0

for subdir in dirs:
    image_total_per_dir = len([name for name in os.listdir(train_set + subdir)])
    total_train_set = total_train_set + image_total_per_dir
    
x_train = np.empty([total_train_set, 227, 227, 3], dtype = 'uint8')

y_train = np.empty([total_train_set, 1], dtype = 'uint8')


for index_dir, subdir in enumerate(dirs):
    image_total_per_dir = len([name for name in os.listdir(train_set + subdir)])
    
    for subdir, dirs, files in os.walk(train_set + subdir):
        for index_file, file in enumerate(files):
            complete_filepath = subdir + os.sep + file
            image = Image.open(complete_filepath)
            if image.format != 'JPG':
                image = image.convert('RGB')
            image_as_array = np.asarray(image)
            image_array_resized = cv2.resize(image_as_array, (227,227), interpolation = cv2.INTER_AREA)
            x_train[current_index] = image_array_resized
            y_train[current_index] = index_dir
            current_index = current_index + 1


y_train = to_categorical(y_train, num_classes = num_classes)

dirs = next(os.walk(test_set))[1]

total_test_set = 0
current_index = 0

for subdir in dirs:
    image_total_per_dir = len([name for name in os.listdir(test_set + subdir)])
    total_test_set = total_test_set + image_total_per_dir
    
x_test = np.empty([total_test_set, 227, 227, 3], dtype = 'uint8')

y_test = np.empty([total_test_set, 1], dtype = 'uint8')

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = value_range,
                                   zoom_range = value_range,
                                   rotation_range = rotation_degree,
                                   width_shift_range = value_shift,
                                   height_shift_range = value_shift,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255,
                                  shear_range = value_range,
                                  zoom_range = value_range,
                                  rotation_range = rotation_degree,
                                  width_shift_range = value_shift,
                                  height_shift_range = value_shift,
                                  horizontal_flip = True)

for index_dir, subdir in enumerate(dirs):
    image_total_per_dir = len([name for name in os.listdir(test_set + subdir)])
    
    for subdir, dirs, files in os.walk(test_set + subdir):
        for index_file, file in enumerate(files):
            complete_filepath = subdir + os.sep + file
            image = Image.open(complete_filepath)
            if image.format != 'JPG':
                image = image.convert('RGB')
            image_as_array = np.asarray(image)
            image_array_resized = cv2.resize(image_as_array, (227,227), interpolation = cv2.INTER_AREA)
            x_test[current_index] = image_array_resized
            y_test[current_index] = index_dir
            current_index = current_index + 1

y_test = to_categorical(y_test, num_classes = num_classes)

training_set = train_datagen.flow(x_train, y_train)
test_set = test_datagen.flow(x_test, y_test)

model = SqueezeNet.SqueezeNet(num_classes, inputs=(227, 227, 3))
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 1000, callbacks = callbacks_without_early_stopping)
