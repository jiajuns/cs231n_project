# !/usr/env python3

'''
Output:
video captioning model itself and produce loss curve

Usage:
main document to train the video captioning model
'''

from __future__ import print_function
import tensorflow as tf
import pickle
# import logging

from model import sequence_2_sequence_LSTM
from embedding_utils import *
from load_feature import load_features, load_captions

# logging.basicConfig(level=logging.INFO)
#=======Change These===============================
tf.app.flags.DEFINE_string("model_name", "baseline_1", "name of the model")
tf.app.flags.DEFINE_integer("state_size", 100, "Size of each model layer.")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Base Learning rate.")
#==================================================

tf.app.flags.DEFINE_float("input_size", 10.0, "input size for each frame")
tf.app.flags.DEFINE_integer("batch_size", 64, "how many videos put per run")
tf.app.flags.DEFINE_integer("max_sentence_length", 200, "maximum captioning sentence length")
tf.app.flags.DEFINE_integer("word_vector_size", 25, "word embedding dimension default is 25 for twitter glove")
tf.app.flags.DEFINE_integer("n_epochs", 10, "number of epoch to run")
FLAGS = tf.app.flags.FLAGS

def load_embedding_dict(file_path):
    if os.path.exists(file_path):
        embedding_dict = pickle.load('file_path')
    else:
        


input_frames = load_feature()
input_captions = load_captions()

dataset_path = curr_path + '/datasets'
word_embedding = 



if __name__ == '__main__':
    model = sequence_2_sequence_LSTM(word_embedding, flags)
    model.build()

    with tf.session() as sess:
        sess.run(tf.global_variables_initializer())
        losses = model.train(sess, train_data)