from __future__ import print_function

import tensorflow as tf
from tqdm import tqdm
import numpy as np

from util import minibatches

class Model(object):
    """Abstracts a Tensorflow graph for a learning task.
    We use various Model classes as usual abstractions to encapsulate tensorflow
    computational graphs. Each algorithm you will construct in this homework will
    inherit from a Model object.
    """

    # def add_placeholders(self):
    #     """Adds placeholder variables to tensorflow computational graph.
    #     Tensorflow uses placeholder variables to represent locations in a
    #     computational graph where data is inserted.  These placeholders are used as
    #     inputs by the rest of the model building and will be fed data during
    #     training.
    #     See for more information:
    #     https://www.tensorflow.org/versions/r0.7/api_docs/python/io_ops.html#placeholders
    #     """
    #     raise NotImplementedError("Each Model must re-implement this method.")

    def create_feed_dict(self, inputs_batch, labels_batch=None):
        """Creates the feed_dict for one step of training.
        A feed_dict takes the form of:
        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }
        If labels_batch is None, then no labels are added to feed_dict.
        Hint: The keys for the feed_dict should be a subset of the placeholder
                    tensors created in add_placeholders.
        Args:
            inputs_batch: A batch of input data.
            labels_batch: A batch of label data.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_embedding_op(self):
        """ Use embedding layer to lookup word vectors
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_prediction_op(self):
        """Implements the core of the model that transforms a batch of input data into predictions.
        Returns:
            pred: A tensor of shape (batch_size, n_classes)
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_loss_op(self, pred):
        """Adds Ops for the loss function to the computational graph.
        Args:
            pred: A tensor of shape (batch_size, n_classes)
        Returns:
            loss: A 0-d tensor (scalar) output
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_training_op(self, loss):
        """Sets up the training Ops.
        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        sess.run() to train the model. See
        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer
        for more information.
        Args:
            loss: Loss tensor (a scalar).
        Returns:
            train_op: The Op for training.
        """

        raise NotImplementedError("Each Model must re-implement this method.")

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        """Perform one step of gradient descent on the provided batch of data.
        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (n_samples, n_features)
            labels_batch: np.ndarray of shape (n_samples, n_classes)
        Returns:
            loss: loss over the batch (a scalar)
        """
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def predict_on_batch(self, sess, inputs_batch):
        """Make predictions for the provided batch of data
        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (n_samples, n_features)
        Returns:
            predictions: np.ndarray of shape (n_samples, n_classes)
        """
        feed = self.create_feed_dict(inputs_batch)
        predictions = sess.run(self.pred, feed_dict=feed)
        return predictions

    def build(self):
        # self.add_placeholders()
        print('start building model ...')
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)

        total_parameter = sum(v.get_shape().num_elements() for v in tf.trainable_variables())
        print('total number of parameter {}'.format(total_parameter))

class sequence_2_sequence_LSTM(Model):

    def __init__(self, embeddings, flags):
        self.pretrained_embeddings = embeddings
        # self.caption_placeholder = video_caption
        self.input_size = flags.input_size
        self.batch_size = flags.batch_size
        self.num_frames = flags.num_frames
        self.max_sentence_length = flags.max_sentence_length
        self.word_vector_size = flags.word_vector_size
        self.n_epochs = flags.n_epochs
        self.hidden_size = flags.hidden_size
        self.learning_rate = flags.learning_rate
        self.dropout_rate = flags.dropout_rate
        self.is_training = flags.is_training

        # ==== set up placeholder tokens ========
        self.frames_placeholder = tf.placeholder(tf.float32, shape=(None, self.num_frames, self.input_size))
        self.caption_placeholder = tf.placeholder(tf.int32, shape=(None, self.max_sentence_length))

    def create_feed_dict(self, input_frames, input_caption=None):
        feed = {
            self.frames_placeholder: input_frames,
        }
        if  input_caption is not None:
            feed[self.caption_placeholder] = input_caption
            
        return feed


    def add_embedding_op(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with tf.variable_scope("embeddings"):
            vec_embeddings = tf.get_variable("embeddings", initializer=self.pretrained_embeddings, trainable=False, dtype=tf.float32)
            if self.caption_placeholder is not None:
                caption_embeddings = tf.nn.embedding_lookup(vec_embeddings, self.caption_placeholder)
            else:
                caption_embeddings = None
        return caption_embeddings

    def add_prediction_op(self):
        """ LSTM encoder and decoder layers
        """
        with tf.variable_scope("LSTM_seq2seq"):
            encoder_output, encoder_state = encoder(input_batch=self.frames_placeholder, 
                                                    hidden_size=self.hidden_size, 
                                                    dropout=self.dropout_rate)

            caption_embeddings = self.add_embedding_op()
            if caption_embeddings is None:
                is_training = False
            else:
                is_training = True
            predict = decoder(encoder_state=encoder_state, 
                                        input_caption=caption_embeddings, 
                                        word_vector_size=self.word_vector_size, 
                                        hidden_size=self.hidden_size, 
                                        max_sentence_length=self.max_sentence_length, 
                                        dropout=self.dropout_rate, train_or_predict = self.is_training)
                                       
            self.predict = predict
            return self.predict

    def add_loss_op(self, word_vecs):
        with tf.variable_scope("loss"):
            caption_embeddings = self.add_embedding_op()
            print('caption embedding shape: ', caption_embeddings.get_shape())
            print('word vecs shape: ', word_vecs.get_shape())
            loss_val = tf.losses.mean_squared_error(caption_embeddings, word_vecs)
        return loss_val

    def add_training_op(self, loss_val):
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.updates = optimizer.minimize(loss_val)

    def train_on_batch(self, sess, input_frames, input_caption):
        feed = self.create_feed_dict(input_frames=input_frames, input_caption=input_caption)
        loss, _, predict= sess.run([self.loss, self.updates, self.predict], feed_dict=feed)
        return loss, predict

    def run_epoch(self, sess, train_data):
        input_frames, captions = train_data         
        # check batch
        batch = minibatches(input_frames, captions, self.batch_size, self.max_sentence_length)
        loss, predict = self.train_on_batch(sess, *batch)
        return loss, predict
    
    def train(self, sess, train_data):
        losses = []
        for epoch in tqdm(range(self.n_epochs)):
            # logger.info("Epoch %d out of %d", epoch + 1, self.config.n_epochs)
            loss, predict = self.run_epoch(sess, train_data)
            losses.append(loss)
        return losses, predict

    def predict_on_batch(self, sess, input_frames):
        feed = self.create_feed_dict(input_frames, None)
        outputs = sess.run([self.predict], feed_dict=feed)
        return outputs


def encoder(input_batch, hidden_size, dropout):
    with tf.variable_scope('encoder') as scope:
        lstm_en_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(hidden_size), output_keep_prob=dropout)
        outputs, state = tf.nn.dynamic_rnn(lstm_en_cell, inputs=input_batch, dtype=tf.float32)
    return outputs, state

def decoder(encoder_state, input_caption, word_vector_size, hidden_size, max_sentence_length, dropout, training = True):
    with tf.variable_scope('decoder') as scope:
        lstm_de_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(hidden_size), output_keep_prob=dropout)
        word_vec_list = []
        state = encoder_state #(N hidden_size)
        
        if train_or_predict == 'train':
            for i in range(max_sentence_length):
                true_word = input_caption[:, i, :]
                if i == 1: scope.reuse_variables() # after the first step reuse varibale
                output_vector, state = lstm_de_cell(true_word, state)
                predict_word = tf.layers.dense(output_vector, units=word_vector_size, name='hidden_to_word')
                word_vec_list.append(predict_word)
        elif train_or_predict == 'test': 
            for i in range(max_sentence_length):
                if i == 0:
                    predict_word = '<START>'
                output_vector, state = lstm_de_cell(predict_word, state)
                predict_word = tf.layers.dense(output_vector, units=word_vector_size, name='hidden_to_word')
                word_vec_list.append(predict_word)
        word_vecs = tf.stack(word_vec_list)
        word_vecs = tf.transpose(word_vecs, perm=[1, 0, 2])
        return word_vecs

