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

train_set = '../dataset/image/train/'
test_set = '../dataset/image/test/'
original_set = '../dataset/image/original/'

classes = next(os.walk(train_set))[1]
num_classes = len(classes)
key_classes = dict((index, category) for index, category in enumerate(classes))

image_width = 227
image_height = 227

data = np.empty([0, image_width, image_height, 3], dtype = 'float32')

train_datagen = ImageDataGenerator(samplewise_center = True,
                                   samplewise_std_normalization = True)

test_datagen = ImageDataGenerator(samplewise_center = True,
                                  samplewise_std_normalization = True)

training_set = train_datagen.flow_from_directory(train_set,
                                                 target_size = (image_width, image_height),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(test_set,
                                                 target_size = (image_width, image_height),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

counter_classes = [0] * num_classes
single_class_size = 800
train_size = single_class_size * num_classes

counter = 0
flag = 1

list_data_classes = [np.empty([single_class_size, image_width, image_height, 3], dtype = 'float32')] * num_classes

x_train = np.empty([single_class_size, image_width, image_height, 3], dtype = 'float32')
y_train = np.empty([single_class_size, 1], dtype = 'float32')

for x, y in training_set:
    flag = 1
    for index in range(0, len(x)):
        array = x[index]
        idx_highest = np.argmax(y[index])
        if( counter_classes[idx_highest] < single_class_size ):
            cur = counter_classes[idx_highest]
            list_data_classes[idx_highest][cur] = array
            y_train[counter] = idx_highest
            counter_classes[idx_highest] += 1
            counter += 1
    for amount in counter_classes:
        print(amount)
        if amount < single_class_size:
            flag = 0
    if flag == 1:
        break


#for generated_x, generated_y in training_set:
#    flag = 1
#    for index in range(0, len(generated_x)):
#        array_to_append = generated_x[index]
#        index_highest = np.argmax(generated_y[index])
#        cat_highest = str(key_classes[index_highest])
#        if(counter_data[index_highest] < limit_each_category):
#            current_pos = counter_data[index_highest]
#            arr[index_highest][current_pos] = array_to_append
#            y_train_zca[data_created] = index_highest
#            counter_data[index_highest] = counter_data[index_highest] + 1
#            data_created = data_created + 1
#    for index, amount in enumerate(counter_data):
#        print(str(classes[index]) + ': ' + str(amount))
#        if amount < limit_each_category:
#            flag = 0
#    if flag == 1:
#        break

y_train_zca = to_categorical(y_train_zca, num_classes = num_classes)    
data_created = 0

for data in arr:
    for el in data:
        x_train_zca[data_created] = el
        data_created = data_created + 1
    

counter_data = [0] * num_classes
limit_each_category = 400
limit_data_augmented = limit_each_category * num_classes

data_created = 0
flag = 1

arr = [np.empty([limit_each_category, image_width, image_height, 3], dtype = 'float32')] * num_classes


x_test_zca = np.empty([limit_data_augmented, image_width, image_height, 3], dtype = 'float32')
y_test_zca = np.empty([limit_data_augmented, 1], dtype = 'uint8')
for generated_x, generated_y in test_set:
    flag = 1
    for index in range(0, len(generated_x)):
        array_to_append = generated_x[index]
        index_highest = np.argmax(generated_y[index])
        cat_highest = str(key_classes[index_highest])
        if(counter_data[index_highest] < limit_each_category):
            current_pos = counter_data[index_highest]
            arr[index_highest][current_pos] = array_to_append
            y_test_zca[data_created] = index_highest
            counter_data[index_highest] = counter_data[index_highest] + 1
            data_created = data_created + 1
    for index, amount in enumerate(counter_data):
        print(str(classes[index]) + ': ' + str(amount))
        if amount < limit_each_category:
            flag = 0
    if flag == 1:
        break

y_test_zca = to_categorical(y_test_zca, num_classes = num_classes)
data_created = 0

for data in arr:
    for el in data:
        x_test_zca[data_created] = el
        data_created = data_created + 1
#
#for generated_x, generated_y in training_set:
#    flag = 1
#    for index in range(0, len(generated_x)):
#        image_to_save = generated_x[index]
#        index_highest = np.argmax(generated_y[index])
#        cat_highest = str(key_classes[index_highest])
#        if(counter_data[index_highest] < limit_each_category):
#            current_pos = counter_data[index_highest]
#            arr[index_highest][current_pos] = index_highest
#            y_train_zca[data_created] = index_highest
##            image_to_save.save(train_set_augmented + cat_highest + '/' + cat_highest + '_' + str(counter_data[index_highest]) + '.png')
##            augmented_flow[index_highest][index] = image_to_save
#            counter_data[index_highest] = counter_data[index_highest] + 1
#            data_created = data_created + 1
#    for amount in counter_data:
#        print(amount)
#        if amount < limit_each_category:
#            flag = 0
#    if flag == 1:
#        break

#
#for data in arr:
#    x_train_zca = np.append(x_train_zca, data, axis = 0)
#    

#y_train_zca = to_categorical(y_train_zca, num_classes = num_classes)
###array generator
        
    
#for generated_x, generated_y in training_set:
#    flag = 1
#    for index in range(0, len(generated_x)):
#        image_to_save = generated_x[index]
#        image_to_save = image_to_save.astype('uint8')
#        image_to_save = Image.fromarray(image_to_save)
#        index_highest = np.argmax(generated_y[index])
#        augmented_flow[index_highest].append
#        cat_highest = str(key_classes[index_highest])
#        directory_to_save = train_set_augmented + cat_highest
#        if not os.path.exists(directory_to_save):
#            os.makedirs(directory_to_save)
#        if(counter_data[index_highest] < limit_each_category):    
#            image_to_save.save(train_set_augmented + cat_highest + '/' + cat_highest + '_' + str(counter_data[index_highest]) + '.png')
#            counter_data[index_highest] = counter_data[index_highest] + 1
#            data_created = data_created + 1
#    for amount in counter_data:
#        print(amount)
#        if amount < limit_each_category:
#            flag = 0
#    if flag == 1:
#        break

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


