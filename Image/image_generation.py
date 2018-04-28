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
from PIL import ImageEnhance
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

original_set = '../dataset/image/original/'

train_set = original_set

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
num_classes = len(dirs)

total_test_set = 0
current_index = 0

for subdir in dirs:
    image_total_per_dir = len([name for name in os.listdir(test_set + subdir)])
    total_test_set = total_test_set + image_total_per_dir
    
x_test = np.empty([total_test_set, 227, 227, 3], dtype = 'uint8')

y_test = np.empty([total_test_set, 1], dtype = 'uint8')


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


dirs = next(os.walk(train_set))[1]

classes = dirs
num_classes = len(dirs)

key_classes = dict((index, category) for index, category in enumerate(classes))

augmented_flow = dict((index, []) for index, category in enumerate(classes))

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

#train_datagen = ImageDataGenerator(rescale = 1./255,
#                                   shear_range = value_range,
#                                   zoom_range = value_range,
#                                   rotation_range = rotation_degree,
#                                   width_shift_range = value_shift,
#                                   height_shift_range = value_shift,
#                                   horizontal_flip = True,
#                                   samplewise_center = True,
#                                   samplewise_std_normalization = True)

train_datagen = ImageDataGenerator(rescale = 1./255,
                                  shear_range = value_range,
                                  zoom_range = value_range,
                                  rotation_range = rotation_degree,
                                  width_shift_range = value_shift,
                                  height_shift_range = value_shift,
                                  horizontal_flip = True,
                                  zca_whitening = True)

test_datagen = ImageDataGenerator(rescale = 1./255,
                                  shear_range = value_range,
                                  zoom_range = value_range,
                                  rotation_range = rotation_degree,
                                  width_shift_range = value_shift,
                                  height_shift_range = value_shift,
                                  horizontal_flip = True,
                                  samplewise_center = True,
                                  samplewise_std_normalization = True)

train_datagen.fit(x_train)

training_set = train_datagen.flow_from_directory(original_set,
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

train_set_augmented = '../dataset/image/training_set_augmented_zca/'

test_set_augmented = '../dataset/image/test_set_augmented_zca/'

counter_data = [0] * num_classes
limit_each_category = 5000
limit_data_augmented = limit_each_category * num_classes

data_created = 0

flag = 1
    
for generated_x, generated_y in training_set:
    flag = 1
    for index in range(0, len(generated_x)):
        image_to_save = generated_x[index]
        image_to_save = Image.fromarray(image_to_save)
        index_highest = np.argmax(generated_y[index])
        cat_highest = str(key_classes[index_highest])
        directory_to_save = train_set_augmented + cat_highest
        if not os.path.exists(directory_to_save):
            os.makedirs(directory_to_save)
        if(counter_data[index_highest] < 5000):    
            image_to_save.save(train_set_augmented + cat_highest + '/' + cat_highest + '_' + str(counter_data[index_highest]) + '.png')
            counter_data[index_highest] = counter_data[index_highest] + 1
            data_created = data_created + 1
            print(data_created)
    for amount in counter_data:
        if amount < 5000:
            flag = 0
    if flag == 1:
        break
#fig = plt.figure(figsize = (10, 10))
#
#columns = 5
#rows = 5
#
#arr_uint8 = x_generated[0]
#arr_uint8 = (arr_uint8 * 255 / np.max(arr_uint8)).astype('uint8')
#img = Image.fromarray(arr_uint8)
#img.save('test.png')
#
#for i in range(1, columns * rows + 1):
#    image = x_generated[i]
#    fig.add_subplot(rows, columns, i)
#    plt.imshow(image)
#
#plt.show()
    

#classifier.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 50, callbacks = callbacks)


