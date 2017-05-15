'''
frame classification

classify each frame of a video 

treat video classification as image classification

sequentially train 10 frames 

'''
import keras
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
from PIL import Image
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras import optimizers
from keras.layers.pooling import MaxPooling3D
from keras.layers.core import Flatten
from keras.utils import to_categorical
from keras import regularizers
import glob
import os
import matplotlib.pyplot as plt 

class frame_classification(object):

    def __init__(self, lr, num_classes,  reg = 0.01, name = 'VGG16', shape = (224, 224, 3)):
        self.lr = lr # learning rate
        self.name = name # model name in case if we use different architecture
        self.size = shape # default size into pretrained model 
        self.hist = None
        self.num_classes = num_classes
        self.reg = reg
    
    def model_create(self):
        
        '''create model'''
        
        if self.name == 'VGG16':

            # build up model
            input = Input(shape=self.size, name = 'image_input')

            # vgg 
            basic_vgg = VGG16(weights='imagenet', include_top=True)
            output_vgg16 = basic_vgg(input)
            score = Dense(self.num_classes, activation='softmax', name='predictions',          
                          kernel_regularizer=regularizers.l2(self.reg))(output_vgg16)
            my_model = Model(inputs=input, outputs=score)

            # Nesterov Momentum
            sgd_m = keras.optimizers.SGD(lr=self.lr, momentum=0.9, decay=1e-6, nesterov=True)
            my_model.compile(loss='categorical_crossentropy',
              optimizer=sgd_m,
              metrics=['accuracy'])
        
        return my_model

    def train(self, X, y, bsize = 32, epoch = 10, verbose = False, split_ratio = 0.2):

        '''
        X: train features data
        y: train data true labels
        bsize: batch_size
        epoch: epoch 
        verbose: show process in training or not 
        split_ratio: validation set split ratio 
        '''

        self.model = self.model_create()
        # to one hot format
        y = to_categorical(y, self.num_classes)

        hist = self.model.fit(X, y, batch_size = bsize, epochs = epoch, verbose = verbose, validation_split = split_ratio)
        self.hist = hist 

    def predict(self, Xte, yte):

        '''
        Xte: test features data
        yte: test data true label
        '''
        ypred = self.model.predict(Xte)
        ypred = np.argmax(ypred, axis = 1)
        print('Test Accuracy: {0}'.format(np.mean(ypred == yte)))

    def plot(self):

        '''
        plot accuracy and loss 
        '''
        plt.figure()
        plt.subplot(121)
        plt.plot(self.hist.history['acc'])
        plt.plot(self.hist.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')

        plt.subplot(122)
        plt.plot(self.hist.history['loss'])
        plt.plot(self.hist.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

