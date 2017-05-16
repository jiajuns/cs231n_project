# !usr/env python 3.5

# import module
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
from PIL import Image
from keras.layers import Dense, GlobalAveragePooling2D, Activation
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras import optimizers
from keras.layers.pooling import MaxPooling3D
from keras.layers.core import Flatten
from keras.utils import to_categorical
import glob
import os
import h5py
import tensorflow as tf
from keras.layers.core import Dropout
from keras import regularizers
from tqdm import tqdm
from scipy.misc import imread, imsave
from sklearn.externals.joblib import Parallel, delayed
from skimage import img_as_float
import multiprocess as mp
import cv2

curr = os.getcwd()
video_dir = curr + '/datasets/frames'

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
        
        if name == 'VGG16':
            self.model = self.vgg_16_pretrained()
        else:
            pass

    def vgg_16_pretrained(self):
        '''
        VGG 16 pretrained model without last 3 fully-connected
        '''

        # vgg without 3 fc
        return VGG16(weights='imagenet', include_top=False)
        

    def load_features(self, frame_dir, num_videos):
        '''
        Concanate video frames over short clip period
        
        frame_dir: frames folder directory
        num_videos: how many videos needed
        '''
        
        all_data = np.zeros((num_videos, 10, 7, 7, 512)) # (#samples, #frames_per_video, h, w, c); h, w, c from vgg output
        labels = np.load(os.getcwd() + '/datasets/category.npy')
        
        for i in tqdm(range(0, num_videos)):
            idx = labels[i, 0]
            features_ls = []
            path = os.path.join(frame_dir, idx)
            
            if not os.path.exists(path):
                print('path not exists for {}'.format(idx))
                continue
            for fn in glob.glob(path+'/*.jpg'):
                im = Image.open(fn)
                # resize image 
                h, w, c= self.size
                im_resized = im.resize((h, w), Image.ANTIALIAS) 

                # transform to array
                im_arr = np.transpose(np.array(im_resized), (0,1,2))

                # preprocess image
                im_arr = np.expand_dims(im_arr, axis=0) # add one dimension as 1
                im_arr.flags.writeable = True
                im_arr = im_arr.astype(np.float64)
                im_arr = preprocess_input(im_arr)

                # output vgg16 without 3 fc layers
                features = self.model.predict(im_arr)
                features_ls.append(features)
                
            # image_paths_list = [path + '/frame' +str(idx)+'.jpg' for idx in range(1, 11, 1)]

            # images = self.process_images(image_paths_list)
            
            # features_ls = [self.model.predict(im) for im in images]
                
            # concatenate
            con_feat = np.concatenate(features_ls, axis = 0)
            all_data[0] = con_feat

        return all_data
    
    def process_images(self, image_paths_list):
        '''
        multiprocess load and process frames
        '''
        
        model = VGG16(weights='imagenet', include_top=False)
        images = Parallel(n_jobs=4, verbose=5)(
                delayed(cv2.imread)(f) for f in image_paths_list
        )
        
        def resize_method():
            def resize(im):
                im_resized = cv2.resize(im, (224, 224))
                im_arr = np.expand_dims(im_resized, axis=0) # add one dimension as 1
                im_arr.flags.writeable = True
                im_arr = im_arr.astype(np.float64)
                im_arr = preprocess_input(im_arr)
                return im_arr
            return resize

        p = mp.Pool(mp.cpu_count())
        images = p.map(resize_method(), images)
        p.close()
        p.join()
        
        images = Parallel(n_jobs=5, verbose=5)(
            delayed(img_as_float)(f) for f in images
        )
        
        return images

    def split_train_test(self):
        '''
        split features and true labels into training sets and test sets
        '''
        data = load_features

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
        x = Input(shape = (10, 7, 7, 512))
        mp = MaxPooling3D(pool_size=(3, 2, 2), strides=(3, 2, 2), padding='valid', data_format='channels_last')(x)
        mp_flat = Flatten()(mp)
        fc1 = Dense(units = 2048, kernel_regularizer=regularizers.l2(reg))(mp_flat)
        
        # fc2 = Dense(units = 512, kernel_regularizer=regularizers.l2(self.reg))(fc1)
        
        fc3 = Dense(units = num_classes, kernel_regularizer=regularizers.l2(reg))(fc1)
        sf = Activation('softmax')(fc3)
        add_model = Model(inputs=x, outputs=sf)
        sgd_m = optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
        add_model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        ytr = to_categorical(ytr, num_classes = num_classes)

        print('Model is Training...')
        hist = add_model.fit(Xtr, ytr, epochs=epochs, batch_size= bsize, validation_split = split_ratio, verbose = verbose)

        self.add_model = add_model
        self.hist = hist

    def predict(self, Xte, yte):
        '''
        Xte: test feature data (sample_size, temporal/frames, height, width, filter_num/channels)
        yte: test true labels (sample_size, )
        '''
        ypred = self.add_model.predict(Xte)
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
        plt.legend(['train', 'validation'], loc='upper left')
        
        plt.subplot(122)
        plt.plot(hist.history['acc'])
        plt.plot(hist.history['val_acc'])
        plt.title('model acc')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
                


