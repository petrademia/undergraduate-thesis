# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 18:39:42 2018

@author: Chastine
"""

from flask import Flask
from flask import request
import base64
import io
from PIL import Image
from PIL.Image import core as _imaging
import numpy as np
from flask.json import jsonify
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
import pymongo
from pymongo import MongoClient
import numpy as np
from numpy.testing import assert_allclose
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, Dense
from keras.callbacks import ModelCheckpoint
from bson.json_util import loads, dumps
import json
from urllib.request import urlopen
import cv2

from twython import Twython, TwythonError
import jsonpickle
import numpy as np
import re
import itertools
from collections import Counter

def load_data_and_labels():
    
#    positive_examples = list(open("./data/rt-polarity.pos", "r", encoding='latin-1').readlines())
#    positive_examples = [s.strip() for s in positive_examples]
#    negative_examples = list(open("./data/rt-polarity.neg", "r", encoding='latin-1').readlines())
#    negative_examples = [s.strip() for s in negative_examples]
    
    banjir_tweets = list(open("dataset/banjir.json", "r").readlines())
    kebakaran_tweets = list(open("dataset/macet.json", "r").readlines())
    macet_tweets = list(open("dataset/kebakaran.json", "r").readlines())
    
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

def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]

def load_data():
    """
    Loads and preprocessed data for the dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels()
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]

from datetime import timedelta
from flask import make_response, request, current_app
from functools import update_wrapper


def crossdomain(origin=None, methods=None, headers=None,
                max_age=21600, attach_to_all=True,
                automatic_options=True):
    if methods is not None:
        methods = ', '.join(sorted(x.upper() for x in methods))
    if headers is not None and not isinstance(headers, list):
        headers = ', '.join(x.upper() for x in headers)
    if not isinstance(origin, list):
        origin = ', '.join(origin)
    if isinstance(max_age, timedelta):
        max_age = max_age.total_seconds()

    def get_methods():
        if methods is not None:
            return methods

        options_resp = current_app.make_default_options_response()
        return options_resp.headers['allow']

    def decorator(f):
        def wrapped_function(*args, **kwargs):
            if automatic_options and request.method == 'OPTIONS':
                resp = current_app.make_default_options_response()
            else:
                resp = make_response(f(*args, **kwargs))
            if not attach_to_all and request.method != 'OPTIONS':
                return resp

            h = resp.headers

            h['Access-Control-Allow-Origin'] = origin
            h['Access-Control-Allow-Methods'] = get_methods()
            h['Access-Control-Max-Age'] = str(max_age)
            if headers is not None:
                h['Access-Control-Allow-Headers'] = headers
            return resp

        f.provide_automatic_options = False
        return update_wrapper(wrapped_function, f)
    return decorator

keyed_classes_image = {
            0: 'Banjir',
            1: 'Kebakaran',
            2: 'Macet'
        }


keyed_classes_text = {
            0: 'Banjir',
            1: 'Macet',
            2: 'Kebakaran'
        }

APP_KEY = 	'qutZDvhBlGhClbKMrx9vlbVmr'
APP_SECRET = '1u3UogRkFgmYkI3JM9reQB54N50BxrcsMFN3GX2wlkQVj2mGxz'
OAUTH_TOKEN = '877015267-PhJ68hXLnEQIJLFPPeMjLh9VhQomqnBfu5P0NwLm'
OAUTH_TOKEN_SECRET = 'vVWTkiDefigEtYMsOcLddotIYlTyxSr5FFGUj2ILrLjSX'

twitter = Twython(APP_KEY, APP_SECRET, OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
twitter.verify_credentials()

print ('Loading data')
x, y, vocabulary, vocabulary_inv = load_data()


embedding_lstm_classifier = load_model('weights/text/lstm/weights@epoch-003-1.00.hdf5')
embedding_weights = embedding_lstm_classifier.get_weights()[0]

from gensim.models import FastText

fasttext_embedding_skipgram = FastText.load_fasttext_format('weights/text/fasttext-skipgram/model')

text_classifier = load_model('weights/text/lstm-without-embedding/weights@epoch-008-1.00.hdf5')

image_classifier = load_model('weights/image/cnn/weights@epoch-006-1.00.hdf5')


#searchQuery = 'banjir -filter:retweets'
#maxTweets = 100
#tweetsPerQuery = 100
#fileName = 'banjir_new.json'
#sinceId = None
#max_id = -1
#tweetCount = 0
#import numpy as np
import re
import itertools
from collections import Counter
#import json
from unicodedata import normalize



def clean_str(string):
    string = string.replace('"', '')
    string = string.replace('\'', '')
    string = string.replace('\\n', ' ')
    string = re.sub(r"\\u[A-Za-z0-9]*", "", string)
    string = ' '.join(re.sub("(@[A-Za-z0-9_]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",string).split())
    string = string.strip('0123456789')
    string = string.lstrip(' ')
    string = string.lstrip('rt')
    string = string.lstrip('RT')
    string = string.lstrip(' ')
    empty = ""
    for el in string:
        if(not el.isdigit()):
            empty = empty+el
    empty = empty.lstrip(' ')
    return empty.strip().lower()

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def predict_image(image_url):
    file = io.BytesIO(urlopen(image_url).read())
    img = Image.open(file)
    if img.format != 'JPG':
        img = img.convert('RGB')
    img_array = np.asarray(img)
    img_resized = cv2.resize(img_array, (227,227), interpolation = cv2.INTER_AREA)
    img_array = np.expand_dims(img_resized, axis = 0)
    prediction = image_classifier.predict(img_array)
    probability = np.amax(prediction)
    index_highest = np.argmax(prediction)
    text_highest = keyed_classes_image[index_highest]
    return (float(probability), int(index_highest), text_highest)

def predict_text(sentence):
    
    padding_word="<PAD/>"
    sequence_length = 30
    to_predict = np.empty((30, 128), dtype=np.float32)
    sentence = sentence.split(" ")
    num_padding = sequence_length - len(sentence)
    new_sentence = sentence + [padding_word] * num_padding
    for index, word in enumerate(new_sentence):
#        print(index)
        if word in vocabulary:
            to_predict[index] = embedding_weights[vocabulary[word]]
        else:
            to_predict[index] = fasttext_embedding_skipgram.wv[word]
            
    to_predict = np.expand_dims(to_predict, axis = 0)
    prediction = text_classifier.predict(to_predict)
    probability = np.amax(prediction)
    index_highest = np.argmax(prediction)
    text_highest = keyed_classes_text[index_highest]
    return (float(probability), int(index_highest), text_highest)

client = MongoClient()
undergrad_thesis = client.undergrad_thesis

tweets = undergrad_thesis.tweets

#text_classifier = load_model('../../Text/lstm/weights@epoch-003-1.00.hdf5')


post = {"test": "data"}

@app.route("/image", methods=['POST'])
def image():
    tweet = request.form['tweet']
    image = request.files['image'].read()
    img = Image.open(io.BytesIO(image))
    img_array = np.asarray(img)
    encoded_img = base64.b64encode(img_array)
#    data = {"image": img_array}
#    tweets.insert_one(data)
    img_array = np.expand_dims(img, axis = 0)
    new_model = load_model("../cnn/weights@epoch-006-1.00.hdf5")
    prediction = new_model.predict(img_array)
    index_highest = np.argmax(prediction)
    return str(index_highest)


@app.route("/image", methods=['GET'])
def retrieve_all_image():
#    all_tweets = tweets.find({}, {'_id': False})
#    all_tweets = list(all_tweets)
    output = []
    for tweet in tweets.find({}, {'_id': False}):
        output.append({'image': tweet['image']})
    return jsonify({'result': output})


@app.route("/text", methods=['POST'])
def predict_single_text():
    sentence = request.form['sentence']
    padding_word="<PAD/>"
    sequence_length = 30
    to_predict = np.empty((30, 128), dtype=np.float32)
    sentence = sentence.split(" ")
    num_padding = sequence_length - len(sentence)
    new_sentence = sentence + [padding_word] * num_padding
    for index, word in enumerate(new_sentence):
#        print(index)
        if word in vocabulary:
            to_predict[index] = embedding_weights[vocabulary[word]]
        else:
            to_predict[index] = fasttext_embedding_skipgram.wv[word]
            
    to_predict = np.expand_dims(to_predict, axis = 0)
    prediction = text_classifier.predict(to_predict)
    index_highest = np.argmax(prediction)
    text_highest = keyed_classes_text[index_highest]
    data = {
            'index_highest': str(index_highest),
            'text_highest': text_highest
            }
    return jsonify(data)
    

@app.route("/crawl", methods=['POST', 'OPTIONS'])
@crossdomain(origin='*')
def crawl():
    
    keyword = request.form['keyword']
    count = request.form['count']
    
    tweets = undergrad_thesis['tweets_'+keyword]
    searchQuery = keyword + ' -filter:retweets'
    maxTweets = int(count)
    tweetsPerQuery = 100
    fileName = 'crawled_' + keyword + '.json'
    sinceId = None
    max_id = -1
    tweetCount = 0
    tweetWithImagesCount = 0
    new_tweets = twitter.search(q=searchQuery, count=tweetsPerQuery)
    
    print('Downloading max {0} tweets'.format(maxTweets))
    
    text = ''
    
    f = open(fileName, 'w+')
    f = f.readlines()
    
    #if f != []:
    #    max_id = int(f[-1].split(',')[0])
    
    #sinceId = 929151725948362752
    
    with open(fileName, 'w+') as f:
        while tweetCount < maxTweets:
            try:
                if(max_id <= 0):
                    if(not sinceId):
#                        print("no max_id and no sinceId")
                        new_tweets = twitter.search(q=searchQuery, count=tweetsPerQuery)
                    else:
#                        print("no max_id but sinceId exist")
                        new_tweets = twitter.search(q=searchQuery, count=tweetsPerQuery, since_id=sinceId)
                else:
                    if (not sinceId):
#                        print("max_id exist but no sinceId")
                        new_tweets = twitter.search(q=searchQuery, count=tweetsPerQuery, max_id=(max_id - 1))
                    else:
#                        print("max_id and sinceId exist")
                        new_tweets = twitter.search(q=searchQuery, count=tweetsPerQuery, max_id=(max_id - 1), since_id=sinceId)
                if not new_tweets:
                    print('No more tweets found')
                    break
                for tweet in new_tweets['statuses']:
                    
                    text = clean_str(tweet['text'])
                    softmax_highest_text_classifier, predicted_text_index, predicted_text_string = predict_text(text)
                    data = {
                            'original_tweet': str(tweet['text']),
                            'processed_tweet': text,
                            'probability_text': softmax_highest_text_classifier,
                            'predicted_text_index': predicted_text_index,
                            'predicted_text_string': predicted_text_string
                            }
                    if 'entities' in tweet:
                        if 'media' in tweet['entities']:
                            tweetWithImagesCount = tweetWithImagesCount + 1
                            url = tweet['entities']['media'][0]['media_url']
                            print(url)
                            softmax_highest_image_classifier, predicted_image_index, predicted_image_string = predict_image(url)
                            data.update({
                                        'probability_image': softmax_highest_image_classifier,
                                        'predicted_image_index': predicted_image_index,
                                        'predicted_image_string': predicted_image_string,
                                        'image_url': url
                                    })
                    tweets.insert_one(data)
                                                       
    #                print(tweet['entities'])
    #                f.write(jsonpickle.encode(tweet['id'], unpicklable=False) + ',' + jsonpickle.encode(tweet['text'], unpicklable=False) + '\n')
                tweetCount += len(new_tweets['statuses'])
                print('Downloaded {0} tweets'.format(tweetCount))
                max_id = new_tweets['statuses'][-1]['id']
            except TwythonError as e:
                print(e.error_code)
                break
    data = {
            'tweet_count': tweetCount,
            'tweet_with_images' : tweetWithImagesCount
            }
    return jsonify(data)

@app.route("/fetch", methods=['POST', 'OPTIONS'])
@crossdomain(origin='*')
def fetch():
    keyword = request.form['keyword']
    tweets = undergrad_thesis['tweets_'+keyword]
    data = []
    for tweet in tweets.find():
        record = {
                'id': str(tweet['_id']),
                'original_tweet': tweet['original_tweet'],
                'processed_tweet': tweet['processed_tweet'],
                'probability_text': tweet['probability_text'],
                'predicted_text_index': tweet['predicted_text_index'],
                'predicted_text_string': tweet['predicted_text_string']
                }
        if 'image_url' in tweet:
            record.update({
                    'probability_image': tweet['probability_image'],
                    'predicted_image_index': tweet['predicted_image_index'],
                    'predicted_image_string': tweet['predicted_image_string'],
                    'image_url': tweet['image_url']
                })
        data.append(record)
    return jsonify(data)
    