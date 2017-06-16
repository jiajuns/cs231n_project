# !/usr/env python3

'''
Output:
id_caption_dict.pickle: id caption mapping 
glove_dic.pickle: word vector mapping 
vocabulary.pickle: filtered words vector mapping 

Usage:
load vocabulary.pickle after running this file 
'''

from __future__ import print_function
import numpy as np 
import os
import wget
import zipfile
import collections
from collections import Counter
from datetime import datetime
import pickle
import matplotlib.pyplot as plt 
import json 
from util import *

def dataLoader(url, filename, path, text_dir):
    '''
    url: (str) data load link 
    filename: (str) data filename with surffix
    path: (str) dataset storage path 
    '''

    # download data
    filepath = os.path.join(path, filename)
    if not os.path.exists(filepath):
        try:
            print('Downloading Data ...')
            wget.download(url, filepath)
        except:
            raise Exception('Downloading Error! Check the url and filepath.')

    elif os.stat(filepath).st_size == 0:
        raise Exception('Data is Empty! Check validation of url.')

    print(" ")
    print('Successfully Downloaded Data {0}!'.format(filename))

    if not os.path.exists(text_dir):
    # unzip file
        print('Unzipping data file...')
        if not zipfile.is_zipfile(filepath):
            raise Exception('No {} file or it is not a zip file'.format(filename))
        else:
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(path)
            print('Successfully unzip {}'.format(filename))
    else:
        print('Already unzip {}'.format(filename))


def dataMapping(caption_dir, glove_dir, save_path, coco_path):
    '''
    caption_dir: (str) caption directory
    glove_dir: (str) glove directory
    save_path: (str) save filter vocabulary path 
    '''
    mapping = collections.defaultdict(list)
    # load id_caption file
    try:
        with open(caption_dir, 'rb') as handle:
            id2caption = pickle.load(handle)
    except:
        raise Exception('No corresponding {}! Need to dump caption pickle file first!'.format(caption_dir))
    
    # load glove_dir
    try:
        with open(glove_dir, 'rb') as handle:
            word2vector = pickle.load(handle)
    except:
        raise Exception('No corresponding {}! Need to dump glove pickle file first'.format(glove_dir))

    caption_words = []
    for key, value in id2caption.items():
        for caption in value:
            #wordls = ' '.join(i for i in value).split(" ")
            wordls = caption.split(" ")
            caption_words.extend(wordls)
    
    coco_words = list(np.load(open(coco_path, 'rb')))
    caption_words = caption_words + coco_words
    caption_words = list(set(caption_words)) # the first is empty
    print('Unique word length: {}'.format(len(caption_words)))

    # mapping
    for word in caption_words:
        word = word.lower()
        if '-' in word: 
            split_w = word.split('-')
            for w in split_w:
                if w not in mapping:
                    mapping[w] = word2vector[w]
        if word in word2vector:
            if word not in mapping:
                mapping[word] = word2vector[word]

    # <start>, <end>, <unk>
    mapping['<START>'] = [0.0] * len(word2vector[word])
    mapping['<unk>'] = word2vector['<unk>']
    mapping['<pad>'] = np.random.normal(np.random.uniform(-0.4,0.4), np.random.uniform(0,1.75), 50)
    mapping['<END>'] = np.random.normal(np.random.uniform(-0.4,0.4), np.random.uniform(0,1.75), 50)

    print('mapping length: {}'.format(len(mapping)))
    print('Dumpping mapping pickle file...')
    with open(os.path.join(save_path, 'word2Vector.pickle'), 'wb') as handle:
            pickle.dump(mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Finishing dumpping!')

def build_id_caption_dict(json_path, caption_path):
    '''
    json_path: (str) video json file path 
    '''
    if not os.path.exists(caption_path):
        path = json_path
        with open(path) as f:
            train = json.load(f)  # keys() -> [u'info', u'videos', u'sentences']

        dictionary = collections.defaultdict(list)

        for s in train['sentences']: # keys() -> [u'caption', u'video_id', u'sen_id']
            dictionary[s['video_id']].append(s['caption'])
        print(len(dictionary.keys()))
        
        with open('id_caption_dict.pickle', 'wb') as handle:
            pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

        try:
            with open('id_caption_dict.pickle', 'rb') as handle:
                b = pickle.load(handle)
        except:
            raise Exception('Not found pickle file!')

        print('Successfully build id-caption dictionary!')
        print('total number of videos is {}'.format(len(b.keys())))
    
    else:
        print('Caption pickle file already exists!')


def build_glove_dict(glove_dir, path):
    '''
    glove_dir: (str) glove text file path 
    path: (str) save path 
    '''
    glove_dic = collections.defaultdict(list)
    storage_path = os.path.join(path, 'glove_dic.pickle')
    if not os.path.exists(storage_path):
        try: 
            tic = datetime.now()
            with open(glove_dir) as f:
                lines = f.readlines()
                for line in lines:
                    vector = []
                    line = line.split(" ")
                    for idx, ele in enumerate(line):
                        if idx == 0:
                            word = ele 
                        else:
                            vector.append(float(ele))
                    glove_dic[word] = vector
            toc = datetime.now()
            print('Word length: {}'.format(len(glove_dic)))
            print('Time for building glove map: {}'.format(toc-tic))
            print('Dumpping pickle file...')
            with open(storage_path, 'wb') as handle:
                pickle.dump(glove_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('Finishing dumpping!')

        except:
            raise Exception('Not found {0}'.format(glove_dir))
    else:
        print('glove pickle already exists')

def plot_word_distribution(caption_dir):
    try:
        with open(caption_dir, 'rb') as handle:
            id2caption = pickle.load(handle)
    except:
        raise Exception('No corresponding {}! Need to dump caption pickle file first!'.format(caption_dir))
    
    caption_len = []
    for key, value in id2caption.items():
        max_len = max(len(i.split(" ")) for i in value)
        caption_len.append(max_len)

    len_dic = Counter(caption_len)
    length = list(len_dic.keys())
    len_count = list(len_dic.values())
    plt.bar(length, len_count, align='center',color='blue', alpha=0.5)
    plt.grid()
    plt.xlabel('Caption max word length in every video', fontsize = 13)
    plt.ylabel('Length count', fontsize = 13)
    plt.title('Caption length distribution', fontsize = 15)
    plt.savefig(os.getcwd() + '/output/Caption_len_distribution.png')
    plt.show()


if __name__ == '__main__':
    glove_url = 'http://nlp.stanford.edu/data/wordvecs/glove.6B.zip'
    glove_filename = "glove.6B.zip"
    curr_path = os.getcwd()
    dataset_path = curr_path + '/datasets'
    glove_text_dir = os.path.join(dataset_path, 'glove.6B.50d.txt')
    caption_dir = os.path.join(dataset_path, 'id_caption_dict_clean.pickle')
    glove_dir = os.path.join(dataset_path, 'glove_dic.pickle')
    json_path = os.path.join(dataset_path, 'train_2017/videodatainfo_2017.json')
    caption_path = os.path.join(dataset_path, 'id_caption_dict_clean.pickle')
    coco_path = os.path.join(dataset_path, 'CoCo_wordLs.npy')

    build_id_caption_dict(json_path, caption_path)
    dataLoader(glove_url, glove_filename, dataset_path, glove_text_dir)
    build_glove_dict(glove_text_dir, dataset_path)
    dataMapping(caption_dir, glove_dir, dataset_path, coco_path)
    
    dataPath = dataset_path + '/'
    build_word_to_index_dict(dataPath)
    build_caption_data_dict(dataPath, maxLen=20)
    #plot_word_distribution(caption_dir) # optional 
