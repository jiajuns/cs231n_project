
# set up
import numpy as np
import pickle
import os


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
    _, frame_num, hwf = input_frames.shape
    input_frames = input_frames.reshape((-1, frame_num, hwf))
    N = len(captions)
    random_index = np.random.choice(N, batch_size, replace = False)
    batch_input_frames = input_frames[video_ind]
    batch_input_captions = np.zeros((len(video_ind), max_len))
    # batch_input_captions = [captions[i][1] for i in random_index]
    count = 0
    for i in random_index:
        batch_input_captions[count] = captions[i][1]
        count += 1
    return (batch_input_frames, batch_input_captions)



def build_word_to_index_dict():
    curPath = os.getcwd()
    dataPath = curPath + "/datasets/"

    w2v = pickle.load(open(dataPath+"word2Vector.pickle", "rb"))
    id_cap_dict = pickle.load(open(dataPath+"id_caption_dict.pickle", "rb"))

    w2v_keys = list(w2v.keys())
    w2v_keys_sorted = sorted(w2v_keys)

    w2ind_dict = {}
    ind2w_dict = {}

    for ind, word in enumerate(w2v_keys_sorted):
        w2ind_dict[word] = ind
        ind2w_dict[ind] = word

    ind2w_dict[None] = "<pad>"

    # check
    print("the index of bardot:  " + str(w2ind_dict['bardot']))
    print("the word of index 1573:  " + str(ind2w_dict[1573]))


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

        # temp code for test
        if len(res) >= 4:
            return res

    n = len(res)
    for j in range(n, maxLen, 1):
        res.append(None)

    return res

def build_caption_data(maxLen = 20):
    curPath = os.getcwd()
    dataPath = curPath + "/datasets/"
    w2ind = pickle.load(open(dataPath+"word2index.pickle", "rb"))
    id_cap_dict = pickle.load(open(dataPath+"id_caption_dict.pickle", "rb"))


    caption_data = []
    for video_id, captionLs in id_cap_dict.items():
        video_id = video_id[5:]

        for caption in captionLs:
            caption_split = ["<START>"] + caption.split()

            # if len(caption_split) > maxLen: # only take captions within maxLen
            #     break

            captionInd = list(caption_to_ind(caption_split, w2ind, maxLen))
            video_caption_pair = tuple([int(video_id), captionInd])
            caption_data.append(video_caption_pair)

    with open(dataPath+'id_captionInd_pairs.pickle', 'wb') as handle:
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

def load_caption_data(sample_size):
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
    captions_train = pickle.load(open(dataPath+"id_captionInd_train.pickle", "rb"))
    captions_test = pickle.load(open(dataPath+"id_captionInd_test.pickle", "rb"))
    input_frames_train = np.load(dataPath + 'Xtrain_all_15frames.npy')[:sample_size]
    input_frames_train = input_frames_train.reshape((-1, 15, 7*7*512))
    input_frames_test = np.load(dataPath + 'Xtest_all_15frames.npy')
    input_frames_test = input_frames_train.reshape((-1, 15, 7*7*512))
    word_dict = pickle.load(open(dataPath + "word2Vector.pickle", "rb"))
    word2Index = pickle.load(open(dataPath + 'word2index.pickle', 'rb'))
    index2Word = pickle.load(open(dataPath + 'index2word.pickle', 'rb'))
    
    return input_frames_train, input_frames_test, captions_train, \
            captions_test, word_dict, word2Index, index2Word

if __name__ == "__main__":
    curPath = os.getcwd()
    dataPath = curPath + "/datasets/"
    # build_word_to_index_dict()
    #build_caption_data(4)
    #check_caption_data(8831)

    ind2w = pickle.load(open(dataPath+"index2word.pickle", "rb"))
    w2ind = pickle.load(open(dataPath+"word2index.pickle", "rb"))
    captions = pickle.load(open(dataPath+"id_captionInd_pairs.pickle", "rb"))

    for key,values in captions:
        if key == 8831:
            print( values, ":     "," ".join([ind2w[i] for i in values ]))


    # input_frames = np.random.randn(100, 15, 7, 7, 512)
    #
    # batch_i, batch_c = minibatches(input_frames, captions, 64)
    #
    # print('batch_c: ', batch_c[0])
    #
    # ind2w = pickle.load(open(dataPath+"index2word.pickle", "rb"))
    #
    # words = []
    # for i in batch_c[0]:
    #     w = ind2w[i]
    #     words.append(w)
    # print('captions: ', ' '.join(i for i in words))
