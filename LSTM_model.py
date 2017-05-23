# !usr/env python 3.5

# keras
from keras.layers import Dense, Activation
from keras.layers import Input, Flatten, Dense
# from keras.models import Model
from keras.layers.recurrent import LSTM
from keras import optimizers
from keras.layers.core import Flatten
from keras.utils import to_categorical
from keras.layers.core import Dropout
from keras import regularizers
from keras.models import Sequential

# system
import os
import numpy as np
from tqdm import tqdm

# image
from scipy.misc import imread, imsave
import matplotlib.pyplot as plt


'''
include_top = True: contain three fully-connected layers and
could be decoded as imagenet class
include_top = False: not contain three fully-connected

VGG16 default size 224 * 224 * 3
'''

class video_classification(object):
    
    def __init__(self, num_classes, num_frames, name='VGG16', shape =(224, 224, 3), optimizer ='Adam', dropout_rate=0.5, reg=0.01):
        self.size = shape
        self.features = None
        self.hist = None
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.dropout_rate = dropout_rate
        self.reg = reg
        self.optimizer = optimizer
        self.build_model()

    def split_train_test(self):
        '''
        split features and true labels into training sets and test sets
        '''
        pass

    def build_model(self):
        '''
        LSTM model
        LSTM(return all state) + LSTM(return all state) + LSTM(last state) + fully-connected + fully_connected -> softmax crossentropy error
        '''
        print('building model...')
        # self.model = Sequential()
        # self.model.add(LSTM(512, return_sequences=True, input_shape=(self.num_frames, 7*7*512)))
        # self.model.add(LSTM(512, return_sequences=True, dropout=self.dropout_rate))
        # self.model.add(LSTM(256, return_sequences=False, dropout=self.dropout_rate))
        # self.model.add(Dense(128, kernel_regularizer=regularizers.l2(self.reg)))
        # self.model.add(Dense(self.num_classes, kernel_regularizer=regularizers.l2(self.reg), activation='softmax'))
        # self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        
        '''
        (1) Build up model based on paper:
        -- Beyond Short Snippets: Deep Networks for Video Classification
        paper link: https://arxiv.org/abs/1503.08909
        
        The architecture is the following: 
        
        five stacked 512 cells LSTM -> softmax for each frames -> linear weighted to score (dense with unit 1)
        
        In the paper, there are four approaches to couple softmax scores for each frames. But the difference between 
        them is less than 1%. Here I choose the one: return max
        
        (2) keras LSTM Reference:
        LSTM source code link: https://github.com/fchollet/keras/blob/master/keras/layers/recurrent.py#L62
        
        -- Some explanations:
        # to stack recurrent layers, you must use return_sequences=True
        # on any recurrent layer that feeds into another recurrent layer.
        
        -- dropout: two choices, dropout and recurrent dropout  
        recurrent dropout paper link: https://arxiv.org/pdf/1603.05118.pdf
        Here I use both.
        
        -- Input shapes:
        3D tensor with shape `(batch_size, timesteps, input_dim)
        (Optional) 2D tensors with shape `(batch_size, output_dim)
        '''
        self.model = Sequential()
        self.model.add(LSTM(512, return_sequences=True, input_shape=(self.num_frames, 7*7*512)))
        self.model.add(LSTM(512, return_sequences=True, recurrent_dropout=self.dropout_rate, dropout = self.dropout_rate))
        self.model.add(LSTM(512, return_sequences=True, recurrent_dropout=self.dropout_rate, dropout = self.dropout_rate))
        self.model.add(LSTM(512, return_sequences=True, recurrent_dropout=self.dropout_rate, dropout = self.dropout_rate))
        self.model.add(LSTM(512, return_sequences=False, recurrent_dropout=self.dropout_rate, dropout = self.dropout_rate))
        self.model.add(Dense(self.num_classes, kernel_regularizer=regularizers.l2(self.reg), activation='softmax'))
        self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, Xtr, ytr, lr = 1e-3, lr_decay = 1e-6,
              bsize = 32, epochs = 100, split_ratio = 0.2, verbose = 0):
        '''
        Xtr: training features data (sample_size, temporal/frames, height, width, filter_num/channels)
        ytr: training true lables (sample_size,)
        lr: learning rate
        reg: regularization
        lr_decay: learning rate decay
        optimizer: optimizer method
        bsize: minibatch size
        epochs: epoch to train
        split_ratio: validation split ratio
        verbose: boolean, show process stdout (1) or not (0)
        '''
        # num_classes = len(np.unique(ytr))
        # to one-hot dimension
        ytr = to_categorical(ytr, num_classes = self.num_classes)
        
        # check Xtrain shape dimension
        # assert len(Xtr.shape) == 4
        
        Xtr = Xtr.reshape((-1, self.num_frames, 7*7*512))
        print('Model is Training...')
        # print('Xtrain shape:' Xtr.shape)
        
        hist = self.model.fit(Xtr, ytr, epochs=epochs,
                             batch_size=bsize, validation_split=split_ratio,
                             verbose=verbose)
        self.hist = hist

    def predict(self, Xte, yte):
        '''
        Xte: test feature data (sample_size, temporal/frames, height, width, filter_num/channels)
        yte: test true labels (sample_size, )
        '''
        ypred = self.model.predict(Xte)
        ypred = np.argmax(ypred, axis = 1)
        
        assert len(ypred) == len(yte)
        
        acc = np.mean(ypred == yte)
        print('Video Classification Accuracy: {0}'.format(acc))

    def plot(self):
        '''
        plot training history accuracy and loss between training sets and validation sets
        '''
        hist = self.hist
        plt.subplot(121)
        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')

        plt.subplot(122)
        plt.plot(hist.history['acc'])
        plt.plot(hist.history['val_acc'])
        plt.title('model acc')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()