# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 13:07:59 2017

@author: Petrus
"""

from keras.layers import Input, Dense, Embedding, merge, Convolution2D, MaxPooling2D, Dropout
from sklearn.cross_validation import train_test_split
from keras.layers.core import Reshape, Flatten
from keras.callbacks import ModelCheckpoint, TensorBoard
from data_helpers import load_data
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv1D, LSTM, Bidirectional
from sklearn.manifold import TSNE


print ('Loading data')
x, y, vocabulary, vocabulary_inv = load_data()

X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42)

sequence_length = x.shape[1]
vocabulary_size = len(vocabulary_inv)
embedding_dim = 256
filter_sizes = [3,4,5]
num_filters = 512
drop = 0.5

nb_epoch = 100
batch_size = 30

filepath="cnn/weights@epoch-{epoch:03d}-{val_acc:.2f}.hdf5"

callback_tensorboard = TensorBoard(log_dir='./Graph', histogram_freq=0,  
          write_graph=True, write_images=True)

callback_checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)

model = Sequential()
model.add(Embedding(vocabulary_size, 128, input_length=sequence_length))
model.add(Conv1D(64, 3, strides=1, activation='relu'))
model.add(Conv1D(64, 4, strides=1, activation='relu'))
model.add(Conv1D(64, 5, strides=1, activation='relu'))
model.add(Flatten())
#model.add((LSTM(128, dropout=0.2, recurrent_dropout=0.2)))
model.add(Dense(3, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=batch_size, epochs=1000, validation_data=(X_test, y_test), callbacks = [callback_tensorboard, callback_checkpoint])
score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)

#weights = model.get_layer(index=1).get_weights()
#tsne = TSNE(n_components = 3, random_state = 0)
#transformed_weights = tsne.fit_transform(weights)

print('Test score:', score)
print('Test accuracy:', acc)

# this returns a tensor
#inputs = Input(shape=(sequence_length,), dtype='int32')
#embedding = Embedding(output_dim=embedding_dim, input_dim=vocabulary_size, input_length=sequence_length)(inputs)
#reshape = Reshape((sequence_length,embedding_dim,1))(embedding)
#
#conv_0 = Convolution2D(num_filters, filter_sizes[0], embedding_dim, border_mode='valid', init='normal', activation='relu', dim_ordering='tf')(reshape)
#conv_1 = Convolution2D(num_filters, filter_sizes[1], embedding_dim, border_mode='valid', init='normal', activation='relu', dim_ordering='tf')(reshape)
#conv_2 = Convolution2D(num_filters, filter_sizes[2], embedding_dim, border_mode='valid', init='normal', activation='relu', dim_ordering='tf')(reshape)
#
#maxpool_0 = MaxPooling2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), border_mode='valid', dim_ordering='tf')(conv_0)
#maxpool_1 = MaxPooling2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1,1), border_mode='valid', dim_ordering='tf')(conv_1)
#maxpool_2 = MaxPooling2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1,1), border_mode='valid', dim_ordering='tf')(conv_2)
#
#merged_tensor = merge([maxpool_0, maxpool_1, maxpool_2], mode='concat', concat_axis=1)
#flatten = Flatten()(merged_tensor)
## reshape = Reshape((3*num_filters,))(merged_tensor)
#dropout = Dropout(drop)(flatten)
#output = Dense(output_dim=2, activation='softmax')(dropout)
#
## this creates a model that includes
#model = Model(input=inputs, output=output)
#
#checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
#adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#
#model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
#model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, callbacks=[checkpoint], validation_data=(X_test, y_test))  # starts training

