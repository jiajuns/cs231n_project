from keras.layers import Activation, Dense, Flatten, Dropout
from keras.optimizers import Adam, SGD

from keras.layers.convolutional import (Conv2D, MaxPooling2D, MaxPooling3D, Conv3D)

from keras.layers.normalization import (BatchNormalization)
from keras.models import Sequential, load_model, model_from_json
from keras import regularizers

from keras.layers.core import Dense, Dropout, Flatten
from keras.models import Sequential
from kerasmodel.kerasmodelzoo.utils.data import download_file, load_np_data

import sys
import os

import numpy as np
import glob
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt

_C3D_WEIGHTS_URL = 'https://www.dropbox.com/s/ypiwalgtlrtnw8b/c3d-sports1M_weights.h5?dl=1'

class models():
    
    
    def __init__(self, model_name, model = None, num_classes = 10, num_frames = 10, size = (64,64,3), reg = 1e-6):
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.model_name = model_name
        self.reg = reg
        
        if model_name == 'c3d':
            h, w, c = size
            self.shape = (num_frames, h, w, c)
            print(self.shape)
            self.model = self.c3d()
            
        elif model_name == 'lstm':
            pass
        else:
            self.model = model
            print("Unknown")
    
        

    def train(self, X_train, y_train, epochs = 100, batch_size = 32, validation_split = 0.2,
              verbose = 1, lr = 1e-5, decay = 0.98):
        
        # optimizer
        sgd_m = SGD(lr= lr, momentum=0.9, decay=decay, nesterov=True)
        adm = Adam(lr= lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.95)
        
        # train
        self.model.compile(loss='categorical_crossentropy', optimizer=adm, metrics=['accuracy'])
        self.history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, \
                                      validation_split = validation_split, verbose = verbose)
        
        
    def save(self, model_path="./saved_model/"):
        model_json = self.model.to_json()
        with open(model_path + self.model_name + '.json', "w") as json_file:
            json_file.write(model_json)
            
        # serialize weights to HDF5
        self.model.save_weights(model_path + self.model_name + ".h5")
        print("Saved model to disk")
        
        
    def load(self, model_path="./saved_model/"):
        # load json and create model
        json_file = open(model_path + self.model_name + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        
        # load weights into new model
        loaded_model.load_weights(model_path + self.model_name + ".h5")
        self.model = loaded_model
        print("Loaded model from disk")
    
    def plot(self):
        plt.figure()
        plt.grid()
        plt.subplot(121)
        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')

        plt.subplot(122)
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')
        plt.show()
       
    '''
    def c3d(weights=True, summary=True):
        c3d_model = Sequential()
        # 1st layer group
        c3d_model.add(Conv3D(64, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv1',
                                subsample=(1, 1, 1),
                                input_shape=(3, 16, 112, 112)))
        c3d_model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                               border_mode='valid', name='pool1'))

        # 2nd layer group
        c3d_model.add(Conv3D(128, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv2',
                                subsample=(1, 1, 1)))
        c3d_model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool2'))

        # 3rd layer group
        c3d_model.add(Conv3D(256, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv3a',
                                subsample=(1, 1, 1)))
        c3d_model.add(Conv3D(256, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv3b',
                                subsample=(1, 1, 1)))
        c3d_model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool3'))

        # 4th layer group
        c3d_model.add(Conv3D(512, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv4a',
                                subsample=(1, 1, 1)))
        c3d_model.add(Conv3D(512, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv4b',
                                subsample=(1, 1, 1)))
        c3d_model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool4'))

        # 5th layer group
        c3d_model.add(Conv3D(512, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv5a',
                                subsample=(1, 1, 1)))
        c3d_model.add(Conv3D(512, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv5b',
                                subsample=(1, 1, 1)))
        c3d_model.add(ZeroPadding3D(padding=(0, 1, 1)))
        c3d_model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool5'))
        c3d_model.add(Flatten())

        # FC layers group
        c3d_model.add(Dense(4096, activation='relu', name='fc6'))
        c3d_model.add(Dropout(.5))
        c3d_model.add(Dense(4096, activation='relu', name='fc7'))
        c3d_model.add(Dropout(.5))
        c3d_model.add(Dense(487, activation='softmax', name='fc8'))

        if weights:
            filepath = download_file('c3d_weights.h5',
                _C3D_WEIGHTS_URL)
            c3d_model.load_weights(filepath)

        if summary:
            print(c3d_model.summary())

        return c3d_model    
    '''
    
    def c3d(self):
        
        reg = self.reg
        model = Sequential()
        model.add(BatchNormalization(input_shape= self.shape))
        
        # conv_1
        conv_1 = Conv3D(64, (3,3,3), strides=(1,2,2))
        model.add(conv_1)
        print(conv_1.output_shape)
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))

        # conv_2 
        conv_2 = Conv3D(128, (3,3,3), padding='same')
        model.add(conv_2)
        print(conv_2.output_shape)
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(1, 2, 2)))

        # conv_3
        conv_3a = Conv3D(256, (3,3,3), padding='same')
        model.add(conv_3a)
        print(conv_3a.output_shape)
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        
        # conv_3
        conv_3b = Conv3D(256, (3,3,3), padding='same')
        model.add(conv_3b)
        print(conv_3b.output_shape)
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
        
        # conv_4
        conv_4a = Conv3D(512, (3,3,3), padding='same')
        model.add(conv_4a)
        print(conv_4a.output_shape)
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        
        # conv_4
        conv_4b = Conv3D(512, (3,3,3), padding='same')
        model.add(conv_4b)
        print(conv_4b.output_shape)
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
        
        
        
        model.add(Flatten())
        # d1
        d1 = Dense(4096, kernel_regularizer=regularizers.l2(reg))
        model.add(d1)
        print (d1.output_shape)
        model.add(Dropout(0.5))
        
        # d1
        d2 = Dense(1024, kernel_regularizer=regularizers.l2(reg))
        model.add(d2)
        print (d2.output_shape)
        model.add(Dropout(0.5))
        
        # d2
        d3 = Dense(self.num_classes, activation='softmax', kernel_regularizer=regularizers.l2(reg))
        model.add(d3)
        print (d3.output_shape)
        print(model.summary())
        return model
    