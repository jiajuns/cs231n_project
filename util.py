
# set up
import numpy as np
import pickle
import os
import random
import time
import sys
from builtins import range

def minibatches(input_frames, captions, batch_size, max_len):
    '''
    Input Args:
    - input_frames: (dict) {raw video id: np.array shape (frame_num, 4096)}
    - captions: (list) list of tuple [(raw video id, captions index)]
    - batch_size: (int) how big the batch is
    - max_len: (int) maximum sentence length

    Produce minibatch data

    Output:
    - batch: (tuple) (batch_input_frames, batch_input_captions)
    '''
    random.seed(231)
    # _, frame_num, hwc = input_frames.shape
    # input_frames = input_frames.reshape((-1, frame_num, hwc))
    num_captions = len(captions)
    indices = np.arange(num_captions)
    random.shuffle(indices)
    
    for minibatch_start in np.arange(0, num_captions, batch_size):
        minibatch_indices = indices[minibatch_start:minibatch_start + batch_size]
        batch_input_captions = np.zeros((len(minibatch_indices), max_len))
        video_ind = np.empty((len(minibatch_indices)))
        for count, i in enumerate(minibatch_indices):
            batch_input_captions[count] = captions[i][1]
            video_ind[count] = captions[i][0]
        batch_input_frames = [input_frames[int(id)] for id in video_ind]
        batch_input_frames = np.stack(batch_input_frames)
        assert len(batch_input_frames.shape) == 3
        # batch_input_frames = input_frames[video_ind]
        yield (video_ind, batch_input_frames, batch_input_captions)
    
    # for _ in range(pieceNum):
    #     random_index = np.random.choice(N, batch_size, replace = False)
    #     video_ind = np.array(random_index)
    #     batch_input_frames = input_frames[video_ind]
    #     batch_input_captions = np.zeros((len(video_ind), max_len))
    #     for count, i in enumerate(random_index):
    #         video_caps = captions[i]
    #         cap_id = random.choice(range(len(video_caps)))
    #         batch_input_captions[count] = video_caps[cap_id]
    #     yield (video_ind, batch_input_frames, batch_input_captions)

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
    print('hihi')
    w2v = pickle.load(open(dataPath+"word2Vector.pickle", "rb"))
    w2v_keys_sorted = sorted(list(w2v.keys()))
    id_cap_dict = pickle.load(open(dataPath+"id_caption_dict_clean.pickle", "rb"))


    w2ind_dict, ind2w_dict= {}, {}
    for ind, word in enumerate(w2v_keys_sorted):
        w2ind_dict[word] = ind
        ind2w_dict[ind] = word
    print(len(w2v_keys_sorted))
    # store
    with open(dataPath+'word2index.pickle', 'wb') as handle:
        pickle.dump(w2ind_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(dataPath+'index2word.pickle', 'wb') as handle:
        pickle.dump(ind2w_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def caption_to_ind(caption_split, w2ind_dict, maxLen = 20):
    res = []
    for i, word in enumerate(caption_split):
        wordInd = w2ind_dict.get(word, w2ind_dict["<unk>"])
        res.append(wordInd)

    # pad the rest length with <unk>
    n = len(res)
    for j in range(n, maxLen, 1):
        res.append(w2ind_dict["<pad>"])
    return res

def cpation_data(dataPath, id_Ls, maxLen = 20):
    w2ind = pickle.load(open(dataPath+"word2index.pickle", "rb"))
    id_cap_dict = pickle.load(open(dataPath+"id_caption_dict_clean.pickle", "rb"))

    caption_data = []
    for count, video_id in enumerate(id_Ls):
        captionLs = [id_cap_dict["video"+str(video_id)]]
        for caption in captionLs:
            caption_split = caption.split()
            if len(caption_split) >= maxLen-2:
                caption_split = caption_split[0:maxLen-2]
                
            caption_split = ["<START>"] + caption_split + ["<END>"]

            captionInd = list(caption_to_ind(caption_split, w2ind, maxLen))
            video_caption_pair = tuple([int(video_id), captionInd])
            caption_data.append(video_caption_pair)
    return caption_data

def build_caption_data_dict(dataPath, maxLen=20):
    x_train_ind = np.load(dataPath+'x_train_ind_above400.npy')
    x_test_ind = np.load(dataPath+'x_test_ind_above400.npy')
    
    caption_data = cpation_data(dataPath, x_train_ind, maxLen)
    with open(dataPath+'id_captionInd_train.pickle', 'wb') as handle:
        pickle.dump(caption_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    caption_data = cpation_data(dataPath, x_test_ind, maxLen)
    with open(dataPath+'id_captionInd_test.pickle', 'wb') as handle:
        pickle.dump(caption_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


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
        # input_frames_train = np.load(dataPath + 'Xtrain_allCap_15frames.npy')
        input_frames_train = pickle.load(open(dataPath + 'input_frames_train.pickle', 'rb'))
        # input_frames_train = input_frames_train.reshape((-1, 15, 4096))[:sample_size]
        # input_frames_train = input_frames_train.reshape((sample_size, 15, 4096))
        word_dict = pickle.load(open(dataPath + "word2Vector.pickle", "rb"))
        word2Index = pickle.load(open(dataPath + 'word2index.pickle', 'rb'))
        index2Word = pickle.load(open(dataPath + 'index2word.pickle', 'rb'))
        return input_frames_train, captions_train, word_dict, word2Index, index2Word
    else:
        captions_test = pickle.load(open(dataPath+"id_captionInd_test.pickle", "rb"))
        # input_frames_test = np.load(dataPath + 'Xtest_allCap_15frames.npy')
        input_frames_test = pickle.load(open(dataPath + 'input_frames_test.pickle', 'rb'))
        # input_frames_test = input_frames_test.reshape((-1, 15, 4096))[:sample_size]
        # input_frames_test = input_frames_test.reshape((sample_size, 15, 4096))
        return input_frames_test, captions_test

    
    
def train_test_split(data, train_test_ratio=0.8):
    '''
    Input Args:
    - data: (tuple) (input_frames, captions)
        -- input_frames: (dict) {raw video id: np.array shape (frame_num, 4096)}
        -- captions: (list) list of tuple [(raw video id, captions index)]
    - train_test_ratio: (float) train test/val split ratio

    train test/validation data split

    Output:
    train_data: (tuple) (input_frames_train, captions_train)
    test_data: (tuple) (input_frames_test/val, captions_test/val)
    '''
    frames, captions = data
    num_samples = len(frames)
    num_train = int(num_samples * train_test_ratio)
    
    vid = np.array(list(frames.keys()))
    np.random.shuffle(vid)
    
    train_indice = vid[:num_train]
    test_indice = vid[num_train:]
    
    train_frames = {}
    test_frames = {}
    train_captions = []
    test_captions = []
        
    for tr_id in train_indice:
        train_frames[tr_id] = frames[tr_id]
        for tu in captions:
            vid, cap = tu
            if vid == tr_id:
                train_captions.append(tu)
    for te_id in test_indice:
        test_frames[te_id] = frames[te_id]
        for tu in captions:
            vid, cap = tu
            if vid == te_id:
                test_captions.append(tu)

    train_data = (train_frames, train_captions)
    test_data = (test_frames, test_captions)

    return train_data, test_data

class Progbar(object):
    """
    Progbar class copied from keras (https://github.com/fchollet/keras/)
    Displays a progress bar.
    # Arguments
        target: Total number of steps expected.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=None, exact=None):
        """
        Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            exact: List of tuples (name, value_for_last_step).
                The progress bar will display these values directly.
        """
        values = values or []
        exact = exact or []

        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far), current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        for k, v in exact:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1]
        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current)/self.target
            prog_width = int(self.width*prog)
            if prog_width > 0:
                bar += ('='*(prog_width-1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.'*(self.width-prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit*(self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if isinstance(self.sum_values[k], list):
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                else:
                    info += ' - %s: %s' % (k, self.sum_values[k])

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width-self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                sys.stdout.write(info + "\n")

    def add(self, n, values=None):
        self.update(self.seen_so_far+n, values)
