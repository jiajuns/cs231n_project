from __future__ import print_function
import time, os, json
import tensorflow as tf
import pickle
import os
import numpy as np
import logging

from util import *
from model.video_caption import sequence_2_sequence_LSTM
from model.image_caption import image_caption_LSTM

from cs231n.coco_utils import load_coco_data, sample_coco_minibatch, decode_captions

'''
Load COCO data. Show the structure of COCO data
'''
#=======Change These===============================
max_train = 400000
epoch = 10
lr = 1e-5
reg = 1e-5
word_vector_size = 50
result_file = 'coco_train_result' + '_lr' + str(lr) + '_reg' + str(reg) + '-2'
save_model_file = 'coco_bestModel' + '_lr' + str(lr) + '_reg' + str(reg) + '-2'
save_model_file2 = 'coco_lastestModel' + '_lr' + str(lr) + '_reg' + str(reg) + '-2'

loadModel = True #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
loadModel_file = 'coco_lastestModel_lr0.001_reg1e-05-1'
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



'''
Load dictionaries
'''
wvector_dim = 50 # hard code 50
curPath = os.getcwd()
dataPath = curPath + "/datasets/"
sample_size = None # None -> return full size of image frames data
is_training = True
_, _, _, word_dict, word2Index, index2Word = load_caption_data(dataPath, sample_size = sample_size, train = is_training)
print('Vocabulary size of word_dict:', len(word_dict.keys()))
print('Vocabulary size of word2Index:', len(word2Index.keys()))
print('Vocabulary size of index2Word:', len(index2Word.keys()))
assert(len(word_dict.keys()) == len(word2Index.keys()) == len(index2Word.keys()))



'''
Convert CoCo caption index into word embedded index, Build embeded caption array
'''
index2Word_coco = data['idx_to_word']
word2Index_coco = data['word_to_idx']

captions_embededInd_train = {}
captions_videosIds = {}
input_frames_train = {}
word_coco_unique = []
for i in range(len(data['train_image_idxs'])):
    image_idx = data['train_image_idxs'][i]
    caption_coco = data['train_captions'][i]
    
    captions_videosIds[i] = image_idx
    
    caption_embeded = []
    for word_cocoInd in caption_coco:
        word_coco = index2Word_coco[word_cocoInd]
                
        if word_coco not in word2Index:
            word_coco_unique.append(word_coco)
            if word_coco == '<NULL>':
                word_embededInd = word2Index['<pad>']
            elif word_coco == '<UNK>':
                word_embededInd = word2Index['<unk>']
            else:
                continue
        else:
            word_embededInd = word2Index[word_coco]
       
        caption_embeded.append(word_embededInd)
        
    captions_embededInd_train[i] = list(caption_embeded)
    input_frames_train[i] = data['train_features'][image_idx].reshape(1, input_size)

# build embeded caption array
voc_size = 15985
word_embedding = word_embedding_array(word_dict, word_vector_size, word2Index)



'''
Build Model
'''
def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session

# define parameters
#=======Change These===============================
state_size = 512
#==================================================
logging.basicConfig(level=logging.INFO)
tf.app.flags.DEFINE_string("model_name", "sequence2sequence", "name of the model")
tf.app.flags.DEFINE_integer("state_size", state_size, "Size of each model layer.")
tf.app.flags.DEFINE_float("input_size", 4096, "input size for each frame")
tf.app.flags.DEFINE_integer("max_sentence_length", 17, "maximum captioning sentence length")
tf.app.flags.DEFINE_integer("word_vector_size", word_vector_size, "word embedding dimension default is 25 for twitter glove")
tf.app.flags.DEFINE_integer("num_frames", 1, "number of frames per video")
#tf.app.flags.DEFINE_integer("voc_size", 6169, "number of vocabulary")
FLAGS = tf.app.flags.FLAGS        

#=======Change These===============================
batch_size = 64
hidden_size = 512
#==================================================
tf.reset_default_graph()
model = image_caption_LSTM(word_embedding, FLAGS, batch_size=batch_size, hidden_size=hidden_size,
        voc_size = voc_size, n_epochs = epoch, lr = lr, reg = reg, save_model_file = save_model_file, save_model_file2 = save_model_file2)
model.train_embedding = False
model.build()

# run training mode
with get_session() as sess:
    saver = tf.train.Saver()
    if loadModel:
        saver.restore(sess, os.getcwd() + "/model/" + loadModel_file + ".ckpt")
#         saver.restore(sess, os.getcwd() + "/model/bestModel.ckpt")
    else:
        sess.run(tf.global_variables_initializer())
    out = model.train(sess, (input_frames_train, captions_embededInd_train), verbose = True)
    
val_loss, tr_loss, tr_pred, val_pred, train_vid, val_vid = out
coco_train_result = {}
coco_train_result['val_loss'] = val_loss
coco_train_result['tr_loss'] = tr_loss
coco_train_result['tr_pred'] = tr_pred
coco_train_result['val_pred'] = val_pred
coco_train_result['train_vid'] = train_vid
coco_train_result['val_vic'] = val_vid

save_path = os.getcwd()
with open('./output/'+ result_file +'.pickle', 'wb') as handle:
            pickle.dump(coco_train_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('Finish save training process information')
print(result_file)
print(save_model_file)
print(save_model_file2)

