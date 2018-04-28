from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, Callback
from keras import backend as K
from keras.engine.topology import Layer
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from time import time

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from quiver_engine import server

from metrics import precision, recall, f1
from custom_callback import TimingCallback
from custom_layer import LocalResponseNormalization
from custom_function import load_obj, save_obj
x_train = np.load('x_train_6class.npy')
y_train = np.load('y_train_6class.npy')
x_test = np.load('x_test_6class.npy')
y_test = np.load('y_test_6class.npy')

input_shape = x_train[0].shape
num_classes = y_train[0].shape[0]

filepath = "weights/cnn/vgg16/weights@epoch-{epoch:03d}-{val_acc:.2f}.hdf5"

tensorboard = TensorBoard(log_dir='./Graph/vgg16-6class', histogram_freq=0, write_graph=True, write_images=True)
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', verbose=1, patience=5)
timing_callback = TimingCallback()

callbacks = [tensorboard, timing_callback]

classifier = Sequential()
classifier.add(Conv2D(16, (3, 3), strides = 1, input_shape = input_shape, padding = 'same', activation = 'relu'))
classifier.add(Conv2D(16, (3, 3), strides = 1, input_shape = input_shape, padding = 'same', activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2), strides = 2))
classifier.add(Conv2D(32, (3, 3), strides = 1, input_shape = input_shape, padding = 'same', activation = 'relu'))
classifier.add(Conv2D(32, (3, 3), strides = 1, input_shape = input_shape, padding = 'same', activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2), strides = 2))
classifier.add(Conv2D(64, (3, 3), strides = 1, input_shape = input_shape, padding = 'same', activation = 'relu'))
classifier.add(Conv2D(64, (3, 3), strides = 1, input_shape = input_shape, padding = 'same', activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2), strides = 2))
classifier.add(Conv2D(128, (3, 3), strides = 1, input_shape = input_shape, padding = 'same', activation = 'relu'))
classifier.add(Conv2D(128, (3, 3), strides = 1, input_shape = input_shape, padding = 'same', activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2), strides = 2))
classifier.add(Conv2D(128, (3, 3), strides = 1, input_shape = input_shape, padding = 'same', activation = 'relu'))
classifier.add(Conv2D(128, (3, 3), strides = 1, input_shape = input_shape, padding = 'same', activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2), strides = 2))
classifier.add(Flatten())
classifier.add(Dense(units = 1024, activation = 'relu'))
classifier.add(Dense(units = 1024, activation = 'relu'))
classifier.add(Dense(units = num_classes, activation = 'softmax'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy', precision, recall, f1])

classifier.summary()

history = classifier.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 50, callbacks = callbacks)

evaluation = history.history
evaluation['training_time'] = timing_callback.logs

save_obj(evaluation, 'vgg_6class')