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
import os
# import logging

from model.basic_model import sequence_2_sequence_LSTM
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

if __name__ == '__main__':
    curPath = os.getcwd()
    dataPath = curPath + "/datasets/train_2017/"
    captions = pickle.load(open(dataPath+"video_caption_pairLs.pickle", "rb"))
    input_frames = np.random.randn(100, 15, 7, 7, 512)
    word_dict = pickle.load(open(dataset_path + "word2Vector.pickle", "rb"))
    word_2_index = pickle.load(open(dataset_path + 'word2index.pickle', 'rb'))


    wvector_dimension = 50
    word_embedding = np.zeros((len(word_dict), wvector_dimension))
    for word, w_vector in word_dict.items():
        word_index = word_2_index[word]
        word_embedding[word_index] = w_vector


    model = sequence_2_sequence_LSTM(word_embedding, flags)
    model.build()

    with tf.session() as sess:
        sess.run(tf.global_variables_initializer())
        losses = model.train(sess, (input_frames, captions))