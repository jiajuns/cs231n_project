from __future__ import print_function
import tensorflow as tf

from model import sequence_2_sequence_LSTM
from embedding_utils import *


if __name__ == '__main__':
    model = sequence_2_sequence_LSTM(word_embedding, flags)
    model.build()

    with tf.session() as sess:
        sess.run(tf.global_variables_initializer())
        losses = model.train(sess, train_data)