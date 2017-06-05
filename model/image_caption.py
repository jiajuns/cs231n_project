from __future__ import print_function

import tensorflow as tf
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from util import *
import os

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
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)

        total_parameter = sum(v.get_shape().num_elements() for v in tf.trainable_variables())
        print('total number of parameter {}'.format(total_parameter))

class image_caption_LSTM(Model):

    def __init__(self, embeddings, flags, batch_size=64, hidden_size=100,
        voc_size = 6169, n_epochs = 50, lr = 1e-3, reg = 1e-4, mode = 'train', save_model_file = 'bestModel'):
        '''
        Input Args:

        - embeddings: (np.array) shape (vocabulary size, word vector dimension)
        - flags: () store a series of hyperparameters
            -- input_size: (int) default pretrained VGG16 output 7*7*512
            -- batch_size: (int) batch size
            -- num_frames: (int) frame number default is 15
            -- max_sentence_length: (int) max word sentence default is 20
            -- voc_size: (int) vocabulary size
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
        # control by flags
        self.input_size = flags.input_size
        self.num_frames = flags.num_frames
        self.max_sentence_length = flags.max_sentence_length
        self.word_vector_size = flags.word_vector_size
        self.state_size = flags.state_size
        
        # control by outsider
        self.pretrained_embeddings = embeddings
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.voc_size = voc_size
        self.n_epochs = n_epochs
        self.learning_rate = lr
        self.reg = reg
        self.train_embedding = False
        self.best_val = float('inf')
        self.mode = mode

        # ==== set up placeholder tokens ========
        self.frames_placeholder = tf.placeholder(tf.float32, shape=(None, self.num_frames, self.input_size))
        self.caption_placeholder = tf.placeholder(tf.int32, shape=(None, self.max_sentence_length))

    def create_feed_dict(self, input_frames, input_caption, is_training=True):

        feed = {
            self.frames_placeholder: input_frames,
            self.caption_placeholder: input_caption
        }

        return feed

    def add_embedding_op(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with tf.variable_scope("embeddings"):
            vec_embeddings = tf.get_variable("embeddings",
                                             initializer=self.pretrained_embeddings,
                                             trainable=self.train_embedding,
                                             dtype=tf.float32)
        return vec_embeddings

    def add_prediction_op(self):
        """ LSTM encoder and decoder layers
        """
        with tf.variable_scope("LSTM_caption"):
            encoder_output, encoder_state = self.encoder()

            caption_embeddings = tf.nn.embedding_lookup(self.embedding, self.caption_placeholder)

            word_ind, batch_loss = self.decoder(encoder_outputs=encoder_output, encoder_out_state = encoder_state, input_caption=caption_embeddings, 
                true_cap = self.caption_placeholder)
            print(word_ind)
            self.word_ind = word_ind
            return batch_loss

    def add_loss_op(self, batch_loss):
        with tf.variable_scope("loss"):
            loss_val = batch_loss
        return loss_val

    def add_training_op(self, loss_val):
        # learning rate decay
        # https://www.tensorflow.org/versions/r0.11/api_docs/python/train/decaying_the_learning_rate
        starter_lr = self.learning_rate
        lr = tf.train.exponential_decay(starter_lr, global_step = 700*self.n_epochs,
                                            decay_steps = 300, decay_rate = 0.9, staircase=True)
        #optimizer = tf.train.AdamOptimizer(lr)

        optimizer = tf.train.RMSPropOptimizer(learning_rate = lr, decay = 0.95, momentum = 0.9)
        self.updates = optimizer.minimize(loss_val)

    def train_on_batch(self, sess, input_frames, input_caption):
        """
        Training model per batch using self.updates
        return loss for that batch and prediction
        """
        feed = self.create_feed_dict(input_frames=input_frames,
                                     input_caption=input_caption,
                                     is_training=True)
        loss, _, predict_index = sess.run([self.loss, self.updates, self.word_ind], feed_dict=feed)
        self.train_pred = predict_index
        return loss

    def test_on_batch(self, sess, input_frames, input_caption):
        """
        Test model and make prediction
        return loss for that batch and prediction
        """
        feed = self.create_feed_dict(input_frames=input_frames,
                                     input_caption=input_caption,
                                     is_training=False)
        loss, predict_index = sess.run([self.loss, self.word_ind], feed_dict=feed)
        self.test_pred = predict_index
        
        return loss

    def predict_on_batch(self, sess, input_frames, batch_captions ):
        feed = {
            self.frames_placeholder: input_frames,
            self.caption_placeholder: batch_captions 
        }
        predict_index = sess.run([self.word_ind], feed_dict=feed)
        return predict_index

    def test(self, sess, valid_data):
        """
        Given validation or test dataset, do not update wegiths
        return the validation or test loss and predicted word vector
        """
        valid_loss = []
        input_frames, captions = valid_data
        for batch in minibatches(input_frames, captions, self.batch_size, self.max_sentence_length):
            vid, inp, cap = batch
            self.val_id = vid
            loss = self.test_on_batch(sess, inp, cap)
            valid_loss.append(loss)
        return np.mean(valid_loss)

    def run_epoch(self, sess, train_data, valid_data, verbose):
        """
        The controller for each epoch training.
        This function will call training_on_batch for training and test for checking validation loss
        train_data = ( frames( dict(vid:) ), captions( tuple(vid, captionIndex) ) )
        """
        train_losses = []
        input_frames, captions = train_data
        prog = Progbar(target=len(captions)//self.batch_size)
        i = 0
        for batch in minibatches(input_frames, captions, self.batch_size, self.max_sentence_length):
            i += 1
            vid, inp, cap = batch
            self.train_id = vid
            train_loss = self.train_on_batch(sess, inp, cap)
            prog.update(i + 1, exact = [("train loss", train_loss)])
            train_losses.append(train_loss)

            # plot batch iteration vs loss figure
        if verbose: plot_loss(train_losses)

        avg_train_loss = np.mean(train_losses)
        dev_loss = self.test(sess, valid_data)
        return dev_loss, avg_train_loss

    def train(self, sess, train_data, verbose = True):
        '''
        train mode
        '''
        val_losses = []
        train_losses = []
        train, validation = train_test_split(train_data, train_test_ratio=0.8)
        prog = Progbar(target=self.n_epochs)
        for i, epoch in enumerate(range(self.n_epochs)):
            dev_loss, avg_train_loss = self.run_epoch(sess, train, validation, verbose)
            if verbose:
                # print epoch results
                prog.update(i + 1, exact = [("train loss", avg_train_loss), ("dev loss", dev_loss)])
                if dev_loss < self.best_val:
                    self.best_val = dev_loss
                    print(" ")
                    print('Validation loss improved, Save Model!')
                else:
                    print(" ")
                    print("Validation loss doesn't improve")
            if dev_loss < self.best_val:
                saver = tf.train.Saver()
                save_path = saver.save(sess, os.getcwd() + "/model/" + save_model_file + ".ckpt")
            val_losses.append(dev_loss)
            train_losses.append(avg_train_loss)
        return val_losses, train_losses, self.train_pred, self.test_pred, self.train_id, self.val_id

    def predict(self, sess, frames_dict, captions_dict):
        """
        Input Args:
        input_frames: (dictionary), {videoId: frames}
        Output:
        list_video_index: (list), [video_id, ...]
        list_predict_index: (list), [[word_index, word_index...],...]
        """
        list_predict_index = []
        list_video_index = []
        
        key_sorted = sorted(captions_dict.keys())
        num_inputs = len(key_sorted)   
           
        for batch_start in tqdm(np.arange(0, num_inputs, self.batch_size)):
            
            keys = key_sorted[batch_start:batch_start+self.batch_size]
            batch_frames = []
            batch_captions = []
            for key in keys:
                list_video_index.append(key)
                batch_frames.append(frames_dict[key])
                batch_captions.append(captions_dict[key])
               
            predict_index = self.predict_on_batch(sess, batch_frames, batch_captions )
            
            for pred in predict_index:
                list_predict_index.append(pred)

        return list_video_index, list_predict_index

    def encoder(self):
        '''
        Input Args:
        input_batch: (tensor) shape (batch_size, frame_num = 15, channels = 4096)
        hidden_size: (int) output vector dimension for each cell in encoder
        dropout: (placeholder variable) dropout probability keep
        max_len: (int) max sentence length
        
        Output:
        outputs: (tensor list) a series of outputs from cells [[batch_size, hidden_size]]
        state: (tensor) [[batch_size, state_size]]
        '''
        input_batch=self.frames_placeholder
        hidden_size=self.hidden_size
        max_len = self.max_sentence_length

        if self.mode == 'train':
            dropout = 0.7
        else:
            dropout = 1
            
        with tf.variable_scope('encoder') as scope:
            
            lstm_en_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(hidden_size, dropout_keep_prob=dropout)   
            outputs, state = tf.nn.dynamic_rnn(lstm_en_cell,
                                               inputs=input_batch,
                                               dtype=tf.float32,
                                               scope=scope)
        return outputs, state

    def decoder(self, encoder_outputs, encoder_out_state, input_caption, true_cap):
        '''
        Input Args:
        encoder_outputs: (list) a series of hidden states (batch_size, hidden_size)
        input_caption: (tensor) after embedding captions (batch_size, T(frame_num), max_len+frame_num)
        tru_cap: (dict) {video id: captions (string)}
        
        Output:
        outputs: (tensor) save decoder cell output vector
        words: (tensor) save word index 
        '''

        embedding = self.embedding
        word_vector_size = self.word_vector_size
        voc_size=self.voc_size
        hidden_size=self.hidden_size
        max_len=self.max_sentence_length
        
        inp_shape = tf.shape(encoder_outputs)
        batch_size, input_len = inp_shape[0], inp_shape[1]
        
        if self.mode == 'train':
            dropout = 0.7
        else:
            dropout = 1

        with tf.variable_scope('decoder') as scope:
            
            #lstm_de_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(hidden_size), output_keep_prob=dropout)
            lstm_de_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(hidden_size, dropout_keep_prob=dropout)
            outputs = []

           
            # initial state
            state = encoder_out_state
            
            prev_ind = None
            words = []
            losses = tf.constant(0, dtype = tf.float32)
            # decoder output words
            for i in range(max_len):

                if i >= 1: scope.reuse_variables()

                # <START>
                if i == 0: prev_vec = tf.ones([tf.shape(input_caption)[0], word_vector_size], tf.float32) 
                
                def f1(): return prev_vec
                def f2(): return input_caption[:, i, :]
                
                if self.mode == 'train':
                    prev_vec = f2()
                else:
                    prev_vec = f1()

                prev_vec = tf.reshape(prev_vec, [batch_size, word_vector_size])
                
                output_vector, state = lstm_de_cell(prev_vec, state)
                
                # scores
                regularizer = tf.contrib.layers.l2_regularizer(scale = 1e-5)
                logits = tf.layers.dense(output_vector, units = voc_size, name = 'hidden_to_scores', kernel_regularizer = regularizer)

                if self.mode == 'train':
                    targets = tf.reshape(true_cap[:, i], [-1])

                    # batch loss
                    batch_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = targets, logits = logits))
                    losses += batch_loss
                    
                  
                    scores = logits
                    # max score word index 
                    prev_ind = tf.argmax(scores, axis = 1)
                    words.append(prev_ind)
                    prev_vec = tf.nn.embedding_lookup(embedding, prev_ind)
                
                else:

                    # inference to get max score len
                    scores = tf.nn.softmax(logits)
                    
                    # max score word index 
                    prev_ind = tf.argmax(scores, axis = 1)
                    words.append(prev_ind)
                    prev_vec = tf.nn.embedding_lookup(embedding, prev_ind)

            # convert to tensor
            words = tf.stack(words)
            words = tf.transpose(words)
            return words, losses / tf.cast(max_len, tf.float32)

def plot_loss(train_losses):
    plt.plot(range(len(train_losses)), train_losses, 'b-')
    plt.grid()
    plt.xlabel('iteration', fontsize = 13)
    plt.ylabel('Train loss', fontsize = 13)
    plt.title('iteration vs loss', fontsize = 15)
    plt.show()
