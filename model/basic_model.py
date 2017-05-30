from __future__ import print_function

import tensorflow as tf
from tqdm import tqdm

from util import minibatches

class Model(object):
    """Abstracts a Tensorflow graph for a learning task.
    We use various Model classes as usual abstractions to encapsulate tensorflow
    computational graphs. Each algorithm you will construct in this homework will
    inherit from a Model object.
    """

    def add_placeholders(self):
        """Adds placeholder variables to tensorflow computational graph.
        Tensorflow uses placeholder variables to represent locations in a
        computational graph where data is inserted.  These placeholders are used as
        inputs by the rest of the model building and will be fed data during
        training.
        See for more information:
        https://www.tensorflow.org/versions/r0.7/api_docs/python/io_ops.html#placeholders
        """
        raise NotImplementedError("Each Model must re-implement this method.")

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
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)


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

        # ==== set up placeholder tokens ========
        self.frames_placeholder = tf.placeholder(tf.float32, shape=(-1, self.num_frames + self.max_sentence_length, self.input_size))
        self.caption_placeholder = tf.placeholder(tf.int32, shape=(-1, self.max_sentence_length, self.word_vector_size))

    def create_feed_dict(self, input_frames, input_caption=None):
        feed = {
            self.frames_placeholder: input_frames,
        }
        if  input_caption is not None:
            feed[self.caption_placeholder] = input_caption
            feed[self.dropout_placeholder] = self.dropout_rate
        else:
            feed[self.dropout_placeholder] = 1
            
        return feed


    def add_embedding_op(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with tf.variable_scope("embeddings"):
            vec_embeddings = tf.get_variable("embeddings", initializer=self.pretrained_embeddings, trainable=False)
            caption_embeddings = tf.nn.embedding_lookup(vec_embeddings, self.caption_placeholder)
        return caption_embeddings

    def add_prediction_op(self):
        """ LSTM encoder and decoder layers
        """

        with tf.variable_scope("LSTM_seq2seq"):
            encoder_output, encoder_state = encoder(input_batch=self.frames_placeholder, 
                                                    hidden_size=self.hidden_size, 
                                                    dropout=self.dropout_placeholder)
            caption_embeddings = self.add_embedding_op()
            predict_word_vecs = decoder(caption_embeddings, ...)

            self.predict = word_vecs
            return word_vecs

    def add_loss_op(self, word_vecs, caption_embeddings):
        with tf.variable_scope("loss"):
            self.loss = tf.losses.mean_squared_error(caption_embeddings, word_vecs)

    def add_training_op(self):
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.updates = optimizer.minimize(self.loss)

    def train_on_batch(self, sess, input_frames, input_caption):
        feed = self.create_feed_dict(input_frames=input_frames, labels_batch=input_caption)
        loss, _ = sess.run([self.loss, self.updates], feed_dict=feed)
        return loss

    def run_epoch(self, sess, train_data):
        losses = []
        input_frames, captions = train_data
        for i, batch in tqdm(enumerate(minibatches(input_frames, captions, self.batch_size))):
            loss = self.train_on_batch(sess, *batch)
            losses.append(loss)
        return losses
    
    def train(self, sess, train_data):
        losses = []
        for epoch in range(self.n_epochs):
            # logger.info("Epoch %d out of %d", epoch + 1, self.config.n_epochs)
            loss = self.run_epoch(sess, train_data)
            losses.append(loss)
        return losses

    def predict_on_batch(self, sess, input_frames):
        feed = self.create_feed_dict(input_frames, None)
        outputs = sess.run([self.predict], feed_dict=feed)

        return outputs


def encoder(input_batch, hidden_size, dropout):
    with tf.variable_scope('encoder') as scope:
        lstm_en_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(hidden_size, scope=scope), output_keep_prob=dropout)
        outputs, state = tf.nn.dynamic_rnn(lstm_en_cell, input=input_batch)
    return outputs, state

def decoder(encoder_state, input_caption, word_vector_size, hidden_size, max_sentence_length, dropout, train_or_predict='train'):
    with tf.variable_scope('decoder') as scope:
        lstm_de_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(hidden_size, scope=scope), output_keep_prob=dropout)
        word_vec_list = []
        state = encoder_state #(N hidden_size)
        if train_or_predict == 'train':
            for i in range(max_sentence_length):
                true_word = tf.gather_nd(input_caption, tf.stack((tf.range(batch_size), i, tf.range(word_vector_size)), axis = 1))
                output_vector, state = lstm_de_cell(true_word, state)
                predict_word = tf.layers.dense(output_vector, units=word_vector_size, reuse=True)
                word_vec_list.append(predict_word)
        elif train_or_predict == 'test': 
            for i in range(max_sentence_length):
                if i == 0:
                    predict_word = '<START>'
                output_vector, state = lstm_de_cell(predict_word, state)
                predict_word = tf.layers.dense(output_vector, units=word_vector_size, reuse=True)
                word_vec_list.append(predict_word)

        word_vecs = tf.stack(word_vec_list)
        return word_vecs

