from __future__ import print_function
import time, os, json
import tensorflow as tf
import pickle
import os
import numpy as np
import logging
import matplotlib.pyplot as plt

from util import *
from model.video_caption import sequence_2_sequence_LSTM
from model.image_caption import image_caption_LSTM

from cs231n.coco_utils import load_coco_data, sample_coco_minibatch, decode_captions
from cs231n.image_utils import image_from_url

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'



#### BUILD MODEL ####
#=======Change These===============================
max_train = 400000
word_vector_size = 50
#==================================================


data = load_coco_data(pca_features = False, max_train = max_train)
input_size = data['train_features'].shape[1]
maxLen = data['train_captions'].shape[1]
wordLs = []

for caption in data['train_captions']:
    for word in caption:
        wordLs.append(word)
voc_size = len(list(set(wordLs)))

# Print out all the keys and values from the data dictionary
for k, v in data.items():
    if type(v) == np.ndarray:
        print(k, type(v), v.shape, v.dtype)
    else:
        print(k, type(v), len(v))
print('\ninput_size:', input_size, ' maxLen:', maxLen, ' voc_size:', voc_size)
print('Finish loading training data!')
####################################



#### CREATE DICTIONARY ####
curPath = os.getcwd()
dataPath = curPath + "/datasets/"

# pick first 100 for debugging purpose
# load data
sample_size = 7000
wvector_dim = 50
is_training = True
_, _, word_dict, word2Index, index2Word = load_caption_data(sample_size, dataPath, train = is_training)



# create word2ind and ind2word dictionary
index2Word_ori = data['idx_to_word']
word2Index_ori = data['word_to_idx']

wordLs = sorted(list(set(wordLs)))
word2Index_coco = {}
index2Word_coco = {}
for i, word_ind in enumerate(wordLs):
    word = index2Word_ori[word_ind]
    word2Index_coco[word] = i
    index2Word_coco[i] = word
print('finish building dictionary')
####################################




####################################

captions_train = {}
captions_corresponding_videoIds = []
input_frames_train = {}

captions_corresponding_videoIds = data['train_image_idxs']
for i, ind in enumerate(data['train_image_idxs']):
    caption = data['train_captions'][i]
    caption_new = []
    for word_ind in caption:
        word = index2Word_coco[word_ind]
        index_new = word2Index['<unk>']
        if word in word2Index:
            index_new = word2Index[word]  
        caption_new.append( index_new )
        
    captions_train[i] = list(caption_new)
    input_frames_train[i] = data['train_features'][ind].reshape(1, input_size)

voc_size = 6169
word_embedding = word_embedding_array(word_dict, word_vector_size, word2Index)
# word_embedding = np.random.randn(voc_size, word_vector_size).astype(np.float32)

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session


# define parameters
logging.basicConfig(level=logging.INFO)
#=======Change These===============================
state_size = 512
#==================================================

tf.app.flags.DEFINE_string("model_name", "sequence2sequence", "name of the model")
tf.app.flags.DEFINE_integer("state_size", state_size, "Size of each model layer.")
tf.app.flags.DEFINE_float("input_size", 4096, "input size for each frame")
tf.app.flags.DEFINE_integer("max_sentence_length", 17, "maximum captioning sentence length")
tf.app.flags.DEFINE_integer("word_vector_size", word_vector_size, "word embedding dimension default is 25 for twitter glove")
tf.app.flags.DEFINE_integer("num_frames", 1, "number of frames per video")
#tf.app.flags.DEFINE_integer("voc_size", 6169, "number of vocabulary")
FLAGS = tf.app.flags.FLAGS        



# build model graph
tf.reset_default_graph()
#=======Change These===============================
batch_size = 64
epoch = 10
lr = 1e-4
hidden_size = 512
#==================================================

model = image_caption_LSTM(word_embedding, FLAGS, batch_size=batch_size, hidden_size=hidden_size,
        voc_size = voc_size, n_epochs = epoch, lr = lr, reg = 1e-4, save_model_file = 'bestModel')

# model = sequence_2_sequence_LSTM(word_embedding, FLAGS, batch_size=batch_size, hidden_size=hidden_size,
#         voc_size = voc_size, n_epochs = epoch, lr = lr, reg = 1e-3, mode = 'train', save_model_file = 'COCO')

model.train_embedding = False
model.build()


# run training mode
with get_session() as sess:
    saver = tf.train.Saver()
#     saver.restore(sess, os.getcwd() + "/model/bestModel.ckpt")
#     saver.restore(sess, os.getcwd() + "/model/lastestModel.ckpt")
    sess.run(tf.global_variables_initializer())
    out = model.train(sess, (input_frames_train, captions_train), verbose = True)

# plot learning curve
plt.plot(range(len(tr_loss)), tr_loss, 'r-', linewidth = 2, label = 'train')
plt.plot(range(len(val_loss)), val_loss, 'b-', linewidth = 2, label = 'validation')
plt.grid()
plt.xlabel('iteration', fontsize = 13)
plt.ylabel('loss', fontsize = 13)
plt.title('iteration vs loss', fontsize = 15)
plt.legend()
plt.savefig(os.getcwd() + '/output/CoCo_image_caption_learning_curve.png')
# plt.savefig(os.getcwd() + '/output/CoCo_seq2seq_learning_curve.png')

