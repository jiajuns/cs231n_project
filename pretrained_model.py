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
import glob
import os
import h5py as h5py
import tensorflow as tf
from keras.layers.core import Dropout

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
        if name == 'VGG16':
            self.model = self.vgg_16_pretrained()
        else:
            pass

    def vgg_16_pretrained(self):
        # build up model
        input = Input(shape=self.size,name = 'image_input')

        # vgg without 3 fc
        basic_vgg = VGG16(weights='imagenet', include_top=False)
        output_vgg16 = basic_vgg(input)
        my_model = basic_vgg
        # add two fc with 4096 units, respectively
        # x = Flatten(name='flatten')(output_vgg16)
        # x = Dense(4096, activation='relu', name='fc1')(x)
        # x = Dense(4096, activation='relu', name='fc2')(x)
        # my_model = Model(input=input, output=x)

        return my_model

    def load_features(self, frame_dir, num_videos):

        
        all_data = np.zeros((num_videos, 10, 7, 7, 512)) # (#samples, #frames_per_video, h, w, c); h, w, c from vgg output
        
        for i in range(0, num_videos):
            features_ls = []
            path = frame_dir +'/video'+str(i)
            if not os.path.exists(path): continue
                
            for fn in glob.glob(path+'/*.jpg'):
                im = Image.open(fn)

                # resize image 
                h, w, c= self.size
                im_resized = im.resize((h, w),Image.ANTIALIAS) # what's this step? resize image to default size

                # transform to array
                im_arr = np.transpose(im_resized, (0,1,2))

                # preprocess image
                im_arr = np.expand_dims(im_arr, axis=0) # add one dimension as 1
                im_arr.flags.writeable = True
                im_arr = im_arr.astype(np.float64)
                im_arr = preprocess_input(im_arr)

                # output vgg16 without 3 fc layers
                features = self.model.predict(im_arr)
                features_ls.append(features)
                
            # concatenate
            con_feat = np.concatenate(features_ls, axis = 0)
            all_data[0] = con_feat

        return all_data

    def split_train_test(self):
        data = load_features

    def train(self, X, y, lr = 1e-3):
        num_classes = len(np.unique(y))
        
        # create new model

        # Temporal max pooling
        x = Input(shape = (10, 7, 7, 512))
        mp = MaxPooling3D(pool_size=(3, 2, 2), strides=(3, 2, 2), padding='valid', data_format='channels_last')(x)
        mp_flat = Flatten()(mp)
        fc1 = Dense(units = 2048)(mp_flat)
        fc2 = Dense(units = 512)(fc1)
        fc3 = Dense(units = num_classes)(fc2)
        sf = Activation('softmax')(fc3)
        add_model = Model(inputs=x, outputs=sf)
        sgd_m = optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
        add_model.compile(optimizer=sgd_m,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        from keras.utils import to_categorical
        
        y = to_categorical(y, num_classes=20)
        
        
        bsize = X.shape[0] // 10
        bsize = 30

        print('Model is Training...')
        hist = add_model.fit(X, y, epochs=100, batch_size= bsize, validation_split = 0.2, verbose = 0)

        self.add_model = add_model
        return hist

    def predict(self, Xte, yte):
        ypred = self.add_model.predict(Xte)
        ypred = np.argmax(ypred, axis = 1)
        acc = np.mean(ypred == yte)
        print('Video Classification Accuracy: {0}'.format(acc))
                


