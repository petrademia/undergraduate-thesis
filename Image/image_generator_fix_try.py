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
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL.Image import core as _imaging
from quiver_engine import server

train_set = '../dataset/image/train_augment/'
test_set = '../dataset/image/test_augment/'

classes = next(os.walk(train_set))[1]
num_classes = len(classes)
key_classes = dict((index, category) for index, category in enumerate(classes))

image_width = 227
image_height = 227

value_shift = 0.1
value_range = 0.2
rotation_degree = 20

train_datagen = ImageDataGenerator(samplewise_center = True,
                                   samplewise_std_normalization = True)

test_datagen = ImageDataGenerator(samplewise_center = True,
                                  samplewise_std_normalization = True)

#train_datagen = ImageDataGenerator(samplewise_center = True,
#                                   samplewise_std_normalization = True,
#                                   zca_whitening = True)
#
#test_datagen = ImageDataGenerator(samplewise_center = True,
#                                  samplewise_std_normalization = True,
#                                  zca_whitening = True)

training_set = train_datagen.flow_from_directory(train_set,
                                                 target_size = (image_width, image_height),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(test_set,
                                                 target_size = (image_width, image_height),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

counter = 0
flag = 1

x_train = np.empty([0, image_width, image_height, 3], dtype = 'float32')
y_train = np.empty([0, 6], dtype = 'float32')

for x, y in training_set:
    x_train = np.append(x_train, x, axis = 0)
    y_train = np.append(y_train, y, axis = 0)
    counter += len(x)
    print(counter)
    if counter > 30000:
        break

x_train = x_train[0:30000]
y_train = y_train[0:30000]

    
counter = 0
    
x_test = np.empty([0, image_width, image_height, 3], dtype = 'float32')
y_test = np.empty([0, 6], dtype = 'float32')

for x, y in test_set:
    x_test = np.append(x_test, x, axis = 0)
    y_test = np.append(y_test, y, axis = 0)
    counter += len(x)
    print(counter)
    if counter > 2400:
        break

x_test = x_test[0:2400]
y_test = y_test[0:2400]

classifier = Sequential()
classifier.add(Conv2D(24, (11, 11), strides = 4, input_shape = (227, 227, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (3, 3), strides = 2))
classifier.add(Conv2D(64, (5, 5), strides = 1, padding = "same", activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (3, 3), strides = 2))
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
classifier.fit(x_train, y_train, validation_data = (x_test, y_test),  epochs = 1000)
#counter_classes = [0] * num_classes
#single_class_size = 800
#train_size = single_class_size * num_classes
#
#
#list_data_classes = [np.empty([single_class_size, image_width, image_height, 3], dtype = 'float32')] * num_classes
#
#x_train = np.empty([train_size, image_width, image_height, 3], dtype = 'float32')
#y_train = np.empty([train_size, 1], dtype = 'float32')
#
#for x, y in training_set:
#    flag = 1
#    for index in range(0, len(x)):
#        array = x[index]
#        idx_highest = np.argmax(y[index])
#        if( counter_classes[idx_highest] < single_class_size ):
#            cur = counter_classes[idx_highest]
#            list_data_classes[idx_highest][cur] = array
#            counter_classes[idx_highest] += 1
#            counter += 1
#    for amount in counter_classes:
#        print(amount)
#        if amount < single_class_size:
#            flag = 0
#    if flag == 1:
#        break
#    
#
#counter = 0
#
#for index, single_class_data in enumerate(list_data_classes):
#    for data in single_class_data:
#        x_train[counter] = data
#        y_train[counter] = index
#        counter += 1
#        
#y_train = to_categorical(y_train, num_classes = num_classes)
