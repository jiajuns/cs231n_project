
# set up
import numpy as np
import pickle
import os
import random
from builtins import range


def minibatches(input_frames, captions, batch_size, max_len):
    '''
    Input Args:
    - input_frames: (np.array) (sample_size, frame_num, 7, 7, 512)
    - captions: (list of tuple) [(video_index, caption)]
    - batch_size: (int) how big the batch is

    Produce minibatch data

    Output:
    - batch: (tuple) (batch_input_frames, batch_input_captions)
    '''
    random.seed(231)
    _, frame_num, hwc = input_frames.shape
    input_frames = input_frames.reshape((-1, frame_num, hwc))
    N = len(input_frames)
    pieceNum = N // batch_size
    print('pieceNum: ',pieceNum)
    for _ in range(pieceNum):
        random_index = np.random.choice(N, batch_size, replace = False)
        video_ind = np.array(random_index)
        batch_input_frames = input_frames[video_ind]
        batch_input_captions = np.zeros((len(video_ind), max_len))
        count = 0
        for i in random_index:
            video_caps = captions[i]
            cap_id = random.choice(range(len(video_caps)))
            batch_input_captions[count] = video_caps[cap_id]
            count += 1
        yield (batch_input_frames, batch_input_captions)

class ind_word_convertor():
    '''
    class ind_word_convertor() provides function to do convertion between
    word and index
    '''

    def __init__(self):
        dataPath = os.getcwd() + "/datasets/"
        self.w2ind = pickle.load(open(dataPath+"word2index.pickle", "rb"))
        self.ind2w = pickle.load(open(dataPath+"index2word.pickle", "rb"))

    def word_to_index(self, word):
        '''
        input: the word
        reutrn: the corresponding index
        '''
        ind = self.w2ind[word]
        return ind

    def index_to_word(self, index):
        '''
        input: index of word
        reutrn: the corresponding word
        '''
        word = self.ind2w[index]
        return word

def build_word_to_index_dict(dataPath):
    w2v = pickle.load(open(dataPath+"word2Vector.pickle", "rb"))
    w2v_keys_sorted = sorted(list(w2v.keys()))
    id_cap_dict = pickle.load(open(dataPath+"id_caption_dict.pickle", "rb"))


    w2ind_dict, ind2w_dict= {}, {}
    for ind, word in enumerate(w2v_keys_sorted):
        w2ind_dict[word] = ind
        ind2w_dict[ind] = word

    # store
    with open(dataPath+'word2index.pickle', 'wb') as handle:
        pickle.dump(w2ind_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(dataPath+'index2word.pickle', 'wb') as handle:
        pickle.dump(ind2w_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def build_caption_data_dict(dataPath):
    x_train_ind = np.load(dataPath+'x_train_ind_above400.npy')
    x_test_ind = np.load(dataPath+'x_test_ind_above400.npy')
    caption_data = cpation_data(dataPath, x_train_ind, 4)
    with open(dataPath+'id_captionInd_train.pickle', 'wb') as handle:
        pickle.dump(caption_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    caption_data = cpation_data(dataPath, x_test_ind, 4)
    with open(dataPath+'id_captionInd_test.pickle', 'wb') as handle:
        pickle.dump(caption_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def cpation_data(dataPath, id_Ls, maxLen = 20):
    w2ind = pickle.load(open(dataPath+"word2index.pickle", "rb"))
    id_cap_dict = pickle.load(open(dataPath+"id_caption_dict.pickle", "rb"))

    caption_data = []
    for count, video_id in enumerate(id_Ls):
        captionLs = id_cap_dict["video"+str(video_id)]
        for caption in captionLs:
            caption_split = ["<START>"] + caption.split()

            if len(caption_split) > maxLen: # only take captions within maxLen
                break

            captionInd = list(caption_to_ind(caption_split, w2ind, maxLen))
            video_caption_pair = tuple([int(count), captionInd])
            caption_data.append(video_caption_pair)
    return caption_data


def word_embedding_array(word_dict, dim, word2Index):
    '''
    Input Args:
    word_dict: (dict) key: word (str) value: vector (list)
    dim: (int) word vector dimension
    word2Index: (dict) key: word (str) value: index (int) word to index mapping

    Output:
    word_embedding: (np.array) shape (V, M)
    V is the size of vocabulary
    M is the dimension of word vector
    '''
    word_embedding = np.zeros((len(word_dict), dim))
    for word, w_vector in word_dict.items():
         if len(w_vector) != 0:
            word_index = word2Index[word]
            word_embedding[word_index] = w_vector
    return word_embedding.astype(np.float32)

def load_caption_data(sample_size, dataPath, train = True):
    '''
    Input Args:
    sample_size: (int) how many samples loaded to train

    Output:
    captions: (list of tuple) captions data, [(video_id, captions (list))]
    input_frames: (np.array) shape (video_training_sample, 15 frames, 7, 7, 512)
    word_dict: (dict) key: word (str) value: vector (list)
    word2Index: (dict) key: word (str) value: index (int) word to index mapping
    index2Word: (dict) key: index (int) value: word (str) index to word mapping
    '''
    if train:
        captions_train = pickle.load(open(dataPath+"id_captionInd_train.pickle", "rb"))
        input_frames_train = np.load(dataPath + 'Xtrain_all_15frames.npy')
        input_frames_train = input_frames_train.reshape((-1, 15, 7, 7, 512))[:sample_size]
        input_frames_train = input_frames_train.reshape((sample_size, 15, 7*7*512))
        word_dict = pickle.load(open(dataPath + "word2Vector.pickle", "rb"))
        word2Index = pickle.load(open(dataPath + 'word2index.pickle', 'rb'))
        index2Word = pickle.load(open(dataPath + 'index2word.pickle', 'rb'))
        return input_frames_train, captions_train, word_dict, word2Index, index2Word
    else:
        captions_test = pickle.load(open(dataPath+"id_captionInd_test.pickle", "rb"))
        input_frames_test = np.load(dataPath + 'Xtest_all_15frames.npy')
        input_frames_test = input_frames_test.reshape((-1, 15, 7, 7, 512))[:sample_size]
        input_frames_test = input_frames_test.reshape((sample_size, 15, 7*7*512))
        return input_frames_test, captions_test

def train_test_split(data, train_test_ratio=0.8):
    num_samples = data[0].shape[0]
    num_train = int(num_samples * train_test_ratio)
    indexes = list(range(num_samples))

    random.shuffle(indexes)
    train_indexes = indexes[:num_train]
    test_indexes = indexes[num_train:-1]

    train_data = (data[0][train_indexes], data[1][train_indexes])
    test_data = (data[0][test_indexes], data[1][test_indexes])

    return train_data, test_data
