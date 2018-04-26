from keras.models import Sequential, load_model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, Callback
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import os
import cv2
from time import time
from PIL import Image
from quiver_engine import server


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
