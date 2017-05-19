# !usr/env python 3.5

# kerase
from keras.layers import Dense, Activation
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras.layers.recurrent import LSTM
from keras import optimizers
from keras.layers.core import Flatten
from keras.utils import to_categorical
from keras.layers.core import Dropout
from keras import regularizers

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

    def __init__(self, name = 'VGG16', shape = (224, 224, 3)):
        self.size = shape
        self.features = None
        self.hist = None

    def split_train_test(self):
        '''
        split features and true labels into training sets and test sets
        '''
        pass

    def train(self, Xtr, ytr, lr = 1e-3, reg = 0.01, lr_decay = 1e-6, optimizer = 'Adam', \
              bsize = 32, epochs = 100, split_ratio = 0.2, verbose = 0):
        '''
        Temporal feature pooling architecture based on pretrained model
        3D max pooling (T, H, W) + fully-connected + fully_connected + softmax

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
        num_classes = len(np.unique(ytr))
        # create new model

        # Temporal max pooling
        LSTM_model = Sequential()
        LSTM_model.add(LSTM(32, input_shape=(10, 64)))

        # x = Input(shape = (10, 7, 7, 512))
        # h1 =
        # mp = MaxPooling3D(pool_size=(3, 2, 2), strides=(3, 2, 2), padding='valid', data_format='channels_last')(x)
        # mp_flat = Flatten()(mp)
        # fc1 = Dense(units = 4096, kernel_regularizer=regularizers.l2(reg))(mp_flat)

        # fc2 = Dense(units = 256, kernel_regularizer=regularizers.l2(reg))(fc1)
        # fc3 = Dense(units = num_classes, kernel_regularizer=regularizers.l2(reg))(fc1)
        # sf = Activation('softmax')(fc3)


        LSTM_model = Model(inputs=x, outputs=sf)
        sgd_m = optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
        LSTM_model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        ytr = to_categorical(ytr, num_classes = num_classes)

        print('Model is Training...')
        hist = LSTM_model.fit(Xtr, ytr, epochs=epochs,
                             batch_size= bsize, validation_split = split_ratio,
                             verbose = verbose)

        self.LSTM_model = LSTM_model
        self.hist = hist

    def predict(self, Xte, yte):
        '''
        Xte: test feature data (sample_size, temporal/frames, height, width, filter_num/channels)
        yte: test true labels (sample_size, )
        '''
        ypred = self.LSTM_model.predict(Xte)
        ypred = np.argmax(ypred, axis = 1)
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
