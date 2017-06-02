from __future__ import print_function

import tensorflow as tf
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from util import *

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
        self.embedding = self.add_embedding_op()
        self.pred, self.pred_index = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)

        total_parameter = sum(v.get_shape().num_elements() for v in tf.trainable_variables())
        print('total number of parameter {}'.format(total_parameter))

class sequence_2_sequence_LSTM(Model):

    def __init__(self, embeddings, flags):
        '''
        Input Args:

        - embeddings: (np.array) shape (vocabulary size, word vector dimension)
        - flags: () store a series of hyperparameters
            -- input_size: (int) default pretrained VGG16 output 7*7*512
            -- batch_size: (int) batch size
            -- num_frames: (int) frame number default is 15
            -- max_sentence_length: (int) max word sentence default is 20
            -- word_vector_size: (int) depend on which vocabulary used
            -- n_epochs: (int) how many epoches to run
            -- hidden_size: (int) hidden state vector size
            -- learning_rate: (float) learning rate

        Placeholder variables:
        - frames_placeholder: (train X) tensor with shape (sample_size, frame_num, input_size)
        - caption_placeholder: (label Y) tensor with shape (sample_size, max_sentence_length)
        - is_training_placeholder: (train mode) tensor int32 0 or 1
        - dropout_placeholder: (dropout keep probability) tensor float32 or 1 for testing
        '''
        self.pretrained_embeddings = embeddings
        self.input_size = flags.input_size
        self.batch_size = flags.batch_size
        self.num_frames = flags.num_frames
        self.max_sentence_length = flags.max_sentence_length
        self.word_vector_size = flags.word_vector_size
        self.vocabulary_size = flags.vocabulary_size
        self.n_epochs = flags.n_epochs
        self.hidden_size = flags.hidden_size
        self.learning_rate = flags.learning_rate

        # ==== set up placeholder tokens ========
        self.frames_placeholder = tf.placeholder(tf.float32, shape=(None, self.num_frames, self.input_size))
        self.caption_placeholder = tf.placeholder(tf.int32, shape=(None, self.max_sentence_length))
        self.is_training_placeholder = tf.placeholder(tf.int32, shape=[])
        self.dropout_placeholder = tf.placeholder(tf.float32, shape = [])

    def create_feed_dict(self, input_frames, input_caption, is_training=True):

        feed = {
            self.frames_placeholder: input_frames,
            self.caption_placeholder: input_caption
        }

        if is_training is True:
            feed[self.is_training_placeholder] = 1
            feed[self.dropout_placeholder] = 0.5
        else:
            feed[self.is_training_placeholder] = 0
            feed[self.dropout_placeholder] = 1

        return feed

    def add_embedding_op(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with tf.variable_scope("embeddings"):
            vec_embeddings = tf.get_variable("embeddings",
                                             initializer=self.pretrained_embeddings,
                                             trainable=False,
                                             dtype=tf.float32)
        return vec_embeddings

    def add_prediction_op(self):
        """ LSTM encoder and decoder layers
        """
        with tf.variable_scope("LSTM_seq2seq"):
            encoder_output, encoder_state = encoder(input_batch=self.frames_placeholder,
                                                    hidden_size=self.hidden_size,
                                                    dropout=self.dropout_placeholder)

            caption_embeddings = tf.nn.embedding_lookup(self.embedding, self.caption_placeholder)

            predict, pword_ls = decoder(encoder_state=encoder_state,
                                        input_caption=caption_embeddings,
                                        embedding = self.embedding,
                                        word_vector_size = self.word_vector_size,
                                        voca_size=self.vocabulary_size,
                                        hidden_size=self.hidden_size,
                                        max_sentence_length=self.max_sentence_length,
                                        dropout=self.dropout_placeholder,
                                        training = self.is_training_placeholder)

            return predict, pword_ls

    def add_loss_op(self, word_vecs):
        with tf.variable_scope("loss"):
            # caption_embeddings = tf.nn.embedding_lookup(self.embedding, self.caption_placeholder)
            captions = tf.one_hot(self.caption_placeholder, self.vocabulary_size)
            # loss_val = tf.losses.mean_squared_error(caption_embeddings, word_vecs)
            print('batch caption shape: ', captions.get_shape())
            print('batch pred shape: ', word_vecs.get_shape())
            loss_val = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = captions, logits = word_vecs))
            print('loss shape: ', loss_val.get_shape())
        return loss_val

    def add_training_op(self, loss_val):

        # learning rate decay
        # https://www.tensorflow.org/versions/r0.11/api_docs/python/train/decaying_the_learning_rate
        starter_lr = self.learning_rate
        lr = tf.train.exponential_decay(starter_lr, global_step = self.n_epochs,
                                           decay_steps = 10, decay_rate = 0.96, staircase=True)
        optimizer = tf.train.AdamOptimizer(lr)
        self.updates = optimizer.minimize(loss_val)

    def train_on_batch(self, sess, input_frames, input_caption):
        """
        Training model per batch using self.updates
        return loss for that batch and prediction
        """
        feed = self.create_feed_dict(input_frames=input_frames,
                                     input_caption=input_caption,
                                     is_training=True)
        loss, _, predict, predict_index, embedding = sess.run([self.loss, self.updates, self.pred, self.pred_index,self.embedding], feed_dict=feed)
        self.train_pred = predict_index
        return loss, predict, embedding

    def test_on_batch(self, sess, input_frames, input_caption):
        """
        Test model and make prediction
        return loss for that batch and prediction
        """
        feed = self.create_feed_dict(input_frames=input_frames,
                                     input_caption=input_caption,
                                     is_training=False)
        loss, predict, predict_index = sess.run([self.loss, self.pred, self.pred_index], feed_dict=feed)
        self.test_pred = predict_index
        return loss, predict

    def test(self, sess, valid_data):
        """
        Given validation or test dataset, do not update wegiths
        return the validation or test loss and predicted word vector
        """
        valid_loss = []
        input_frames, captions = valid_data
        for batch in minibatches(input_frames, captions, self.batch_size, self.max_sentence_length):
            loss, _ = self.test_on_batch(sess, *batch)
            valid_loss.append(loss)
        return np.mean(valid_loss)

    def run_epoch(self, sess, train_data, valid_data, verbose):
        """
        The controller for each epoch training.
        This function will call training_on_batch for training and test for checking validation loss
        """
        train_losses = []
        input_frames, captions = train_data
        for batch in minibatches(input_frames, captions, self.batch_size, self.max_sentence_length):
            inp, cap = batch
            train_loss, _, embedding = self.train_on_batch(sess, *batch)
            train_losses.append(train_loss)

            # plot batch iteration vs loss figure
        if verbose: plot_loss(train_losses)

        avg_train_loss = np.mean(train_losses)
        dev_loss = self.test(sess, valid_data)
        return dev_loss, avg_train_loss, embedding

    def train(self, sess, train_data, verbose = True):
        '''
        train mode
        '''
        val_losses = []
        train_losses = []
        train, validation = train_test_split(train_data, train_test_ratio=0.8)
        prog = Progbar(target=self.n_epochs)
        for i, epoch in enumerate(range(self.n_epochs)):
            dev_loss, avg_train_loss, embedding = self.run_epoch(sess, train, validation, verbose)
            if verbose:
                # print epoch results
                prog.update(i + 1, exact = [("train loss", avg_train_loss), ("dev loss", dev_loss)])
            val_losses.append(dev_loss)
            train_losses.append(avg_train_loss)
        return val_losses, train_losses, self.train_pred, self.test_pred, embedding

    # def predict_on_batch(self, sess, input_frames):
    #     feed = self.create_feed_dict(input_frames, None)
    #     outputs = sess.run([self.pred], feed_dict=feed)
    #     return outputs

def plot_loss(train_losses):
    plt.plot(range(len(train_losses)), train_losses, 'b-')
    plt.grid()
    plt.xlabel('iteration', fontsize = 13)
    plt.ylabel('Train loss', fontsize = 13)
    plt.title('iteration vs loss', fontsize = 15)
    plt.show()

def encoder(input_batch, hidden_size, dropout):
    with tf.variable_scope('encoder') as scope:
        lstm_en_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(hidden_size), output_keep_prob=dropout)
        outputs, state = tf.nn.dynamic_rnn(lstm_en_cell,
                                           inputs=input_batch,
                                           dtype=tf.float32,
                                           scope=scope)
    return outputs, state

def decoder(encoder_state, input_caption, word_vector_size, embedding, voca_size, hidden_size, max_sentence_length, dropout, training):
    with tf.variable_scope('decoder') as scope:
        lstm_de_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(hidden_size), output_keep_prob=dropout)
        word_vec_list = []
        state = encoder_state #(N hidden_size)

        pword_ls = []
        for i in range(max_sentence_length):
            if i == 0:
                predict_word = tf.zeros([tf.shape(input_caption)[0], word_vector_size], tf.float32)
            def f1(): return predict_word
            def f2(): return input_caption[:, i, :]
            true_word = tf.cond(training < 1, lambda: f1(), lambda: f2())
            if i == 1: scope.reuse_variables()
            output_vector, state = lstm_de_cell(true_word, state)
            scores = tf.layers.dense(output_vector, units=voca_size, name='hidden_to_word')
            pword = tf.argmax(scores, axis = 1)
            pword_ls.append(tf.identity(pword))
            predict_word = tf.nn.embedding_lookup(embedding, pword)
            word_vec_list.append(tf.identity(scores))

        word_vecs = tf.stack(word_vec_list)
        word_vecs = tf.transpose(word_vecs, perm=[1, 0, 2])

        pword_ls = tf.stack(pword_ls)
        pword_ls = tf.transpose(pword_ls)
        return word_vecs, pword_ls

