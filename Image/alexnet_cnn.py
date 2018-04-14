# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 03:35:10 2018

@author: Chastine
"""

#from keras.models import Sequential
#from keras.layers import Conv2D
#from keras.layers import MaxPooling2D
#from keras.layers import Flatten
#from keras.layers import Dense
#from keras.layers import Dropout
#from keras.layers.normalization import BatchNormalization
#from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
#from keras import backend as K
#import matplotlib.pyplot as plt
#from imutils import paths
import keras
import numpy as np
#from quiver_engine import server
import os
from PIL import Image

#print("[INFO] loading images...")
#data = []
#labels = []
#
#imagePaths = sorted(list(paths.list_images('dataset/banjir_train')))
#
#from PIL import Image
#import os, os.path
#
#imgs = []
#path = "./dataset/"
#valid_images = [".jpg", ".gif", ".png", ".tga"]
#
#p = os.listdir(path)
#
#for f in os.listdir(path):
#    print(f)

#from PIL import Image
#import glob
#image_list = []
#
#for filename in glob.glob('./banjir_test/*.png'):
#    im=Image.open(filename)
#    image_list.append(im)
#    print(filename)

train_set = '../dataset/training_set'

x_train = np.empty([12000, 227, 227, 3], dtype = 'uint8')

y_train = np.empty([12000, 1], dtype = 'uint8')

y_train[0:4000] = 0
y_train[4000:8000] = 1
y_train[8000:12000] = 2

x = 0

start = 0

counter = 0

for subdir, dirs, files in os.walk(train_set):
    for index, file in enumerate(files):
        if( counter > 3999):
            break
        filepath = subdir + os.sep + file
        image = Image.open(filepath)
        image_array = np.asarray(image)
        x_train[x] = image_array
        counter = counter + 1
        x = x + 1
    counter = 0

print('done creating training variable')
    
test_set = '../dataset/test_set'

x_test = np.empty([3000, 227, 227, 3], dtype = 'uint8')

y_test = np.empty([3000, 1], dtype = 'uint8')

y_test[0:1000] = 0
y_test[1000:2000] = 1
y_test[2000:3000] = 2

x = 0

start = 0

counter = 0

for subdir, dirs, files in os.walk(test_set):
    for index, file in enumerate(files):
        if( counter > 999):
            break
        filepath = subdir + os.sep + file
        image = Image.open(filepath)
        image_array = np.asarray(image)
        x_test[x] = image_array
        counter = counter + 1
        x = x + 1
    counter = 0

print('done creating testing variable')
    
    
from keras.models import Sequential, load_model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras import backend as K
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
from quiver_engine import server

one_hot_labels = keras.utils.to_categorical(y_train, num_classes=3)

y_train = one_hot_labels

one_hot_labels = keras.utils.to_categorical(y_test, num_classes=3)

y_test = one_hot_labels

filepath = "weights/cnn/alexnet/weights@epoch-{epoch:03d}-{val_acc:.2f}.hdf5"
shift = 0.1

tensorboard = TensorBoard(log_dir='./Graph', histogram_freq=0,  
          write_graph=True, write_images=True)

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)

early_stopping = EarlyStopping(monitor='val_loss', verbose=1, patience=5)

callbacks = [tensorboard, checkpoint, early_stopping]

classifier = Sequential()

classifier.add(Conv2D(96, (11, 11), strides = 4, input_shape = (227, 227, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (3, 3), strides = 2))

classifier.add(BatchNormalization())

classifier.add(Conv2D(256, (5, 5), strides = 1, padding = "same", activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (3, 3), strides = 2))

classifier.add(BatchNormalization())

classifier.add(Conv2D(256, (3, 3), strides = 1, padding = "same", activation = 'relu'))

classifier.add(Conv2D(256, (3, 3), strides = 1, padding = "same", activation = 'relu'))

classifier.add(Conv2D(256, (3, 3), strides = 1, padding = "same", activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (3, 3), strides = 2))

classifier.add(Flatten())

classifier.add(Dropout(0.5))

classifier.add(Dense(units = 1024, activation = 'relu'))

classifier.add(Dense(units = 1024, activation = 'relu'))


classifier.add(Dense(units = 3, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

#from keras.preprocessing.image import ImageDataGenerator
#
#train_datagen = ImageDataGenerator(rescale = 1./255,
#                                   shear_range = 0.2,
#                                   zoom_range = 0.2,
#                                   rotation_range = 20,
#                                   width_shift_range = shift,
#                                   height_shift_range = shift,
#                                   horizontal_flip = True)
#
#test_datagen = ImageDataGenerator(rescale = 1./255,
#                                  shear_range = 0.2,
#                                  zoom_range = 0.2,
#                                  rotation_range = 20,
#                                  width_shift_range = shift,
#                                  height_shift_range = shift,
#                                  horizontal_flip = True)
#
#training_set = train_datagen.flow_from_directory('dataset/training_set',
#                                                 target_size = (227, 227),
#                                                 batch_size = 32,
#                                                 class_mode = 'binary', save_to_dir = './train_augment', shuffle = False)
#
#test_set = test_datagen.flow_from_directory('dataset/test_set',
#                                            target_size = (227, 227),
#                                            batch_size = 32,
#                                            class_mode = 'binary', save_to_dir = './test_augment', shuffle = False)

#
#def layer_to_visualize(layer):
#    inputs = [K.learning_phase()] + classifier.inputs
#
#    _convout1_f = K.function(inputs, [layer.output])
#    def convout1_f(X):
#        # The [0] is to disable the training phase flag
#        return _convout1_f([0] + [X])
#
#    convolutions = convout1_f(img_to_visualize)
#    convolutions = np.squeeze(convolutions)
#
#    print ('Shape of conv:', convolutions.shape)
#
#    n = convolutions.shape[0]
#    n = int(np.ceil(np.sqrt(n)))
#
#    # Visualization of each filter of the layer
#    fig = plt.figure(figsize=(12,8))
#    for i in range(len(convolutions)):
#        ax = fig.add_subplot(n,n,i+1)
#        ax.imshow(convolutions[i], cmap='gray')


classifier.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 1000, batch_size = 32, callbacks = callbacks)

#server.launch(classifier, input_folder = './imgs', classes = ['macet', 'kebakaran', 'banjir'], temp_folder = './tmp')
#
#new_model = load_model("./cnn/weights@epoch-006-1.00.hdf5")
#
#single_image = x_train[15000];
#single_image = np.expand_dims(single_image, axis = 0)
#predict = new_model.predict(single_image)
#index_highest = np.argmax(predict)
