from keras.models import Sequential, load_model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
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

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1(y_true, y_pred):
    
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
    
    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall))

class TimingCallback(Callback):
    def on_train_begin(self, logs={}):
        self.logs = []
        
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = time()
        
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(time() - self.starttime)
    
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

image_size = 227

image_width = 227

image_height = 227

value_shift = 0.1
value_range = 0.2
rotation_degree = 20

tensorboard = TensorBoard(log_dir='./Graph/one_fourth_alexnet', histogram_freq=0,  
          write_graph=True, write_images=True)
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_acc', verbose=1, patience=5)
timing_callback = TimingCallback()

callbacks = [tensorboard, checkpoint, early_stopping, timing_callback]

#callbacks = [tensorboard, checkpoint]

#callbacks = [tensorboard, checkpoint, early_stopping]

train_set = '../dataset/image/train_augment/'

test_set = '../dataset/image/test_augment/'

file_to_write = open('log.txt', mode='w+')

dirs = next(os.walk(train_set))[1]

dirs.remove('objek')
dirs.remove('orang')
dirs.remove('teks')


classes = dirs
num_classes = len(dirs)

total_train_set = 0
current_index = 0

image_width = 227
image_height = 227

for subdir in dirs:
    image_total_per_dir = len([name for name in os.listdir(train_set + subdir)])
    total_train_set = total_train_set + image_total_per_dir
    
x_train = np.empty([total_train_set, image_width, image_height, 3], dtype = 'uint8')

y_train = np.empty([total_train_set, 1], dtype = 'uint8')


for index_dir, subdir in enumerate(dirs):
    print('Converting image from ' + subdir + ' folder to array...')
    image_total_per_dir = len([name for name in os.listdir(train_set + subdir)])
    
    for subdir, dirs, files in os.walk(train_set + subdir):
        for index_file, file in enumerate(files):
            complete_filepath = subdir + os.sep + file
            image = Image.open(complete_filepath)
            if image.format != 'JPG':
                image = image.convert('RGB')
            image_as_array = np.asarray(image)
            image_array_resized = cv2.resize(image_as_array, (image_width,image_height), interpolation = cv2.INTER_AREA)
            x_train[current_index] = image_array_resized
            y_train[current_index] = index_dir
            current_index = current_index + 1


y_train = to_categorical(y_train, num_classes = num_classes)



dirs = next(os.walk(test_set))[1]
dirs.remove('objek')
dirs.remove('orang')
dirs.remove('teks')

total_test_set = 0
current_index = 0

for subdir in dirs:
    image_total_per_dir = len([name for name in os.listdir(test_set + subdir)])
    total_test_set = total_test_set + image_total_per_dir
    
x_test = np.empty([total_test_set, image_width, image_height, 3], dtype = 'uint8')

y_test = np.empty([total_test_set, 1], dtype = 'uint8')


for index_dir, subdir in enumerate(dirs):
    print('Converting image from ' + subdir + ' folder to array...')
    image_total_per_dir = len([name for name in os.listdir(test_set + subdir)])
    
    for subdir, dirs, files in os.walk(test_set + subdir):
        for index_file, file in enumerate(files):
            complete_filepath = subdir + os.sep + file
            image = Image.open(complete_filepath)
            if image.format != 'JPG':
                image = image.convert('RGB')
            image_as_array = np.asarray(image)
            image_array_resized = cv2.resize(image_as_array, (image_width,image_height), interpolation = cv2.INTER_AREA)
            x_test[current_index] = image_array_resized
            y_test[current_index] = index_dir
            current_index = current_index + 1

y_test = to_categorical(y_test, num_classes = num_classes)

classifier = Sequential()

classifier.add(Conv2D(24, (11, 11), strides = 4, input_shape = (image_width, image_height, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (3, 3), strides = 2))
classifier.add(LocalResponseNormalization())
classifier.add(Conv2D(64, (5, 5), strides = 1, padding = "same", activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (3, 3), strides = 2))
classifier.add(LocalResponseNormalization())
classifier.add(Conv2D(96, (3, 3), strides = 1, padding = "same", activze = (3, 3), strides = 2))
classifier.add(LocalResponseNormalization())
classifier.add(Conv2D(64, (5, 5), strides = 1, padding = "same", activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (3, 3), strides = 2))
classifier.add(LocalResponseNormalization())
classifier.add(Conv2D(96, (3, 3), strides = 1, padding = "same", actation = 'relu'))
classifier.add(Conv2D(96, (3, 3), strides = 1, padding = "same", activation = 'relu'))
classifier.add(Conv2D(64, (3, 3), strides = 1, padding = "same", activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (3, 3), strides = 2))
classifier.add(Flatten())
classifier.add(Dropout(0.5))
classifier.add(Dense(units = 1024, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units = 1024, activation = 'relu'))
classifier.add(Dense(units = num_classes, activation = 'softmax'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy', precision, recall, f1])

#train_datagen = ImageDataGenerator(zca_whitening = True)
#
#test_datagen = ImageDataGenerator(zca_whitening = True)
#
#train = train_datagen.flow_from_directory(train_set,
#                                      target_size = (227, 227),
#                                      batch_size = 32,
#                                      class_mode = 'categorical')
#
#
#test = test_datagen.flow_from_directory(test_set,
#                                        target_size = (227, 227),
#                                        batch_size = 32,
#                                        class_mode = 'categorical')


#
#test_datagen = ImageDataGenerator(rescale = 1./255,
#                                  shear_range = value_range,
#                                  zoom_range = value_range,
#                                  rotation_range = rotation_degree,
#                                  width_shift_range = value_shift,
#                                  height_shift_range = value_shift,
#                                  horizontal_flip = True,
#                                  samplewise_center = True,
#                                  samplewise_std_normalization = True)
#
#training_set = train_datagen.flow_from_directory('../dataset/image/training_set',
#                                                 target_size = (227, 227),
#                                                 batch_size = 32,
#                                                 class_mode = 'categorical')
#
#test_set = test_datagen.flow_from_directory('../dataset/image/test_set',
#                                            target_size = (227, 227),
#                                            batch_size = 32,
#                                            class_mode = 'categorical')

#training_set = train_datagen.flow(x_train, y_train)
#test_set = test_datagen.flow(x_test, y_test)

#training_set = (x_train, y_train)
#test_set = (x_test, y_test)
#
#batch_size = 32
#
#classifier.fit_generator(train,
#                         steps_per_epoch = 4800/batch_size,
#                         epochs = 100,
#                         validation_steps = 2400/batch_size, validation_data = test, callbacks = callbacks)

history = classifier.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 50, callbacks = callbacks)

#y_test_non_categorized = [np.argmax(single_test) for single_test in y_test]
#
#y_pred = classifier.predict(x_test)
#
#y_pred_non_categorized = [np.argmax(single_pred) for single_pred in y_pred]
#y_pred = classifier.predict(x_test)
#
#y_pred_non_categorized = [np.argmax(single_pred) for single_pred in y_pred]
#
#from sklearn.metrics import confusion_matrix
#conf_mat = confusion_matrix(y_test_non_categorized, y_pred_non_categorized)


