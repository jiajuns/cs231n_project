
# set up
import numpy as np 
import pickle
import os 


def minibatch(input_frames, captions, batch_size):
    '''
    Input: 
    - input_frames: (np.array) (sample_size, frame_num, 7, 7, 512)
    - captions: (list of tuple) [(video_index, caption)]
    - batch_size: (int) how big the batch is 

    Produce minibatch data 

    Output:
    - batch: (tuple) (batch_input_frames, batch_input_captions)
    '''
    np.random.seed(231)
    _, frame_num, oh, ow, of = input_frames.shape
    input_frames = input_frames.reshape((-1, frame_num, oh*ow*of))
    N = len(captions)
    random_index = np.random.choice(N, batch_size, replace = False)
    video_ind = [total_captions[i][1] for i in random_index]
    batch_input_frames = input_frames[video_ind]

    return (batch_input_frames, batch_input_captions)

def index_word_dict(vocDir):
    '''
    Input: 
    - vocDir (str) vocabulary pickle directory

    Output:
    - dict (dictionary) word to index and index to word
    '''
    try:


def index2Word(wordls, vocabulary):
    '''
    Input: 
    - wordls: (list) a list of word 
    - vocabulary: (pickle file)

    Index to Word 

    Output:
    - indexls: (list) indice for word list
    '''
    try: 
        voc = pickle.dump()
    l = len(vocabulary)
    dic = {word: index for word, index in zip(voc.keys(), range(l))}


def word2Index(wordls, vocabulary):



def build_caption_data():
    pass