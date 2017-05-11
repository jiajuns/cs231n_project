# !usr/env python 3.5

# import module
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
import glob
import os

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
        features_ls = []

        # (num_samples, num_frames_per_video, h, w, c)
        # h, w, c are from vgg output
        all_data = np.zeros((5, 10, 7, 7, 512))
        for i in range(0, num_videos):
            for fn in glob.glob(frame_dir +'/video'+str(i)+'/*.jpg'):
                im = Image.open(fn)

                # resize image 
                h, w, c= self.size
                im_resized = im.resize((h, w),Image.ANTIALIAS)

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

    def train(self, X, y):
        # create new model

        # Temporal max pooling
        num_classes = len(np.unique(y))

        # num_frames_per_video
        x = Input(shape = (10, 7, 7, 512))
        mp = MaxPooling3D(pool_size=(2, 2, 2), strides=(1, 1, 1), padding='valid', data_format=None)(x)
        mp_flat = Flatten()(mp)
        fc1 = Dense(units = 2048)(mp_flat)
        fc2 = Dense(units = 512)(fc1)
        fc3 = Dense(units = num_classes)(fc2)
        add_model = Model(inputs=x, outputs=fc2)
        sgd_m = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        add_model.compile(optimizer=sgd_m,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        bsize = X.shape[0] / 20

        print('Model is Training...')
        add_model.fit(X, y, epochs=10, batch_size= bsize, verbose = 1, validation_split = 0.2)

        self.add_model = add_model

    def predict(self, Xte, yte):
        ypred = self.add_model.predict(Xte)
        acc = np.mean(ypred == yte)
        print('Video Classification Accuracy: {0}'.format(acc))
                


