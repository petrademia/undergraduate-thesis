# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 11:23:21 2018

@author: Chastine
"""

from keras.layers import Input, Dense, Embedding, merge, Convolution2D, MaxPooling2D, Dropout
from sklearn.cross_validation import train_test_split
from keras.layers.core import Reshape, Flatten
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from data_helpers import load_data
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv1D, LSTM, Bidirectional
from sklearn.manifold import TSNE
from keras.preprocessing.text import text_to_word_sequence
from keras.models import Model
from keras.preprocessing.text import Tokenizer


print ('Loading data')
x, y, vocabulary, vocabulary_inv = load_data()

X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42)

text_classifier = load_model('lstm/weights@epoch-003-1.00.hdf5')

filepath="lstm-only/weights@epoch-{epoch:03d}-{val_acc:.2f}.hdf5"

tensorboard = TensorBoard(log_dir='./Graph', histogram_freq=0,  
          write_graph=True, write_images=True)

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)

early_stopping = EarlyStopping(monitor='val_loss', verbose=1, patience=5)

callbacks = [tensorboard, checkpoint, early_stopping]


def make_keras_tokenizer():
    banjir_tweets = list(open("dataset-cutted/banjir.json", "r").readlines())
    kebakaran_tweets = list(open("dataset-cutted/macet.json", "r").readlines())
    macet_tweets = list(open("dataset-cutted/kebakaran.json", "r").readlines())
    
    x_text = banjir_tweets + kebakaran_tweets + macet_tweets
    
    t = Tokenizer()
    t.fit_on_texts(x_text)
    
    return t

def make_vocabulary():
    banjir_tweets = list(open("dataset-cutted/banjir.json", "r").readlines())
    kebakaran_tweets = list(open("dataset-cutted/macet.json", "r").readlines())
    macet_tweets = list(open("dataset-cutted/kebakaran.json", "r").readlines())
    
    x_text = banjir_tweets + kebakaran_tweets + macet_tweets
    x_text = [s.split(" ") for s in x_text] 
    
    return x_text

def load_data_and_labels():
    
#    positive_examples = list(open("./data/rt-polarity.pos", "r", encoding='latin-1').readlines())
#    positive_examples = [s.strip() for s in positive_examples]
#    negative_examples = list(open("./data/rt-polarity.neg", "r", encoding='latin-1').readlines())
#    negative_examples = [s.strip() for s in negative_examples]
    
    banjir_tweets = list(open("dataset-cutted/banjir.json", "r").readlines())
    kebakaran_tweets = list(open("dataset-cutted/macet.json", "r").readlines())
    macet_tweets = list(open("dataset-cutted/kebakaran.json", "r").readlines())
    
    banjir_tweets = banjir_tweets[0:50000]
    kebakaran_tweets = kebakaran_tweets[0:50000]
    macet_tweets = macet_tweets[0:50000]
    
    # Split by words
    
    x_text = banjir_tweets + kebakaran_tweets + macet_tweets
    x_text = [s.split(" ") for s in x_text] 
    
#    x_text = positive_examples + negative_examples
#    x_text = [clean_str(sent) for sent in x_text]
#    x_text = [s.split(" ") for s in x_text]
    
    # Generate labels
    
    banjir_labels = [[1, 0, 0] for _ in banjir_tweets]
    kebakaran_labels = [[0, 1, 0] for _ in kebakaran_tweets]
    macet_labels = [[0, 0, 1] for _ in macet_tweets]
    
    y = np.concatenate([banjir_labels, kebakaran_labels, macet_labels], 0)

#    positive_labels = [[0, 1] for _ in positive_examples]
#    negative_labels = [[1, 0] for _ in negative_examples]
#    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

vocabulary = make_vocabulary()

tokenizer = make_keras_tokenizer()

layer_name = 'embedding_2'
intermediate_layer_model = Model(inputs=text_classifier.input,
                                 outputs=text_classifier.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(X_train)

for x in X_train[11]:
    print(vocabulary_inv[x])
    
embedding_weights = text_classifier.get_weights()[0]

model = Sequential()
#model.add(Embedding(vocabulary_size, 128, input_length=sequence_length))
#model.add(Conv1D(64, 3, strides=1, activation='relu'))
#model.add(Conv1D(64, 4, strides=1, activation='relu'))
#model.add(Conv1D(64, 5, strides=1, activation='relu'))
#model.add(Flatten())
model.add((LSTM(128, dropout=0.2, recurrent_dropout=0.2, batch_input_shape=(None, 30, 128))))
model.add(Dense(3, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(intermediate_output, y_train, batch_size=32, epochs=1000, callbacks = callbacks)
#trained_model = load_model('lstm/weights@epoch-003-0.99.hdf5')
#
#new_data = np.reshape(X_test[0], [1, 30])
#
#prediction = trained_model.predict(new_data)
#
#weights = trained_model.get_layer(index = 1).get_weights()
#tsne = TSNE(n_components = 2, verbose = 1)
#transformed_weights = tsne.fit_transform(weights)
#
#weights_zero = trained_model.get_layer(index = 3).get_weights()
