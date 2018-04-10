# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
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


filepath = "cnn/weights@epoch-{epoch:03d}-{val_acc:.2f}.hdf5"
shift = 0.1

tensorboard = TensorBoard(log_dir='./Graph', histogram_freq=0,  
          write_graph=True, write_images=True)

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)

early_stopping = EarlyStopping(monitor='val_loss', verbose=1, patience=5)

callbacks = [tensorboard, checkpoint, early_stopping]

classifier = Sequential()

classifier.add(Conv2D(48, (11, 11), strides = 4, input_shape = (227, 227, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (3, 3), strides = 2))

classifier.add(BatchNormalization())

classifier.add(Conv2D(128, (5, 5), strides = 1, padding = "same", activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (3, 3), strides = 2))

classifier.add(BatchNormalization())

classifier.add(Conv2D(192, (3, 3), strides = 1, padding = "same", activation = 'relu'))

classifier.add(Conv2D(192, (3, 3), strides = 1, padding = "same", activation = 'relu'))

classifier.add(Conv2D(128, (3, 3), strides = 1, padding = "same", activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (3, 3), strides = 2))

classifier.add(Flatten())

classifier.add(Dropout(0.5))

classifier.add(Dense(units = 256, activation = 'relu'))

classifier.add(Dense(units = 256, activation = 'relu'))


classifier.add(Dense(units = 3, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   rotation_range = 20,
                                   width_shift_range = shift,
                                   height_shift_range = shift,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255,
                                  shear_range = 0.2,
                                  zoom_range = 0.2,
                                  rotation_range = 20,
                                  width_shift_range = shift,
                                  height_shift_range = shift,
                                  horizontal_flip = True)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (227, 227),
                                                 batch_size = 32,
                                                 class_mode = 'binary', save_to_dir = './train_augment', shuffle = False)

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (227, 227),
                                            batch_size = 32,
                                            class_mode = 'binary', save_to_dir = './test_augment', shuffle = False)

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


classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 1,
                         validation_data = test_set,
                         validation_steps = 2000)


#server.launch(classifier, input_folder = './imgs', classes = ['macet', 'kebakaran', 'banjir'], temp_folder = './tmp')
