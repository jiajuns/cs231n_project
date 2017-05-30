
# set up
import numpy as np
import pickle
import os


def minibatches(input_frames, captions, batch_size, max_len):
    '''
    Input:
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
    # video_ind = list(set([captions[i][0] for i in random_index]))
    video_ind = range(100)
    batch_input_frames = input_frames[video_ind]
    print('max len: ', max_len)
    batch_input_captions = np.zeros((len(video_ind), max_len))
    # batch_input_captions = [captions[i][1] for i in random_index]
    count = 0
    for i in random_index:
        batch_input_captions[count] = captions[i][1]
        count += 1
    return (batch_input_frames, batch_input_captions)



def build_word_to_index_dict(dataPath):
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
    w2ind_dict["<pad>"] = None

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

def build_caption_data(dataPath, id_Ls, maxLen = 20):
    w2ind = pickle.load(open(dataPath+"word2index.pickle", "rb"))
    id_cap_dict = pickle.load(open(dataPath+"id_caption_dict.pickle", "rb"))


    caption_data = []
    for count, video_id in enumerate(id_Ls):
        captionLs = id_cap_dict["video"+str(video_id)]
        for caption in captionLs:
            caption_split = ["<START>"] + caption.split()

            # if len(caption_split) > maxLen: # only take captions within maxLen
            #     break

            captionInd = list(caption_to_ind(caption_split, w2ind, maxLen))
            video_caption_pair = tuple([int(count), captionInd])
            caption_data.append(video_caption_pair)
    return caption_data

def check_caption_data(dataPath, caption_id):
    caption_data = pickle.load(open(dataPath+"id_captionInd_pairs.pickle", "rb"))
    ind2w = pickle.load(open(dataPath+"index2word.pickle", "rb"))

    print((caption_data[caption_id]))
    for ind in caption_data[caption_id][1]:
        if ind==None:continue
        print (ind2w[ind])



if __name__ == "__main__":
    curPath = os.getcwd()
    dataPath = curPath + "/datasets/"
    # build_word_to_index_dict(dataPath)
    # build_caption_data(4)
    # check_caption_data(dataPath, 4000)

    ind2w = pickle.load(open(dataPath+"index2word.pickle", "rb"))
    w2ind = pickle.load(open(dataPath+"word2index.pickle", "rb"))
    captions = pickle.load(open(dataPath+"id_captionInd_pairs.pickle", "rb"))
    x_train_ind = np.load(dataPath+'x_train_ind_above400.npy')
    x_test_ind = np.load(dataPath+'x_test_ind_above400.npy')

    caption_data = build_caption_data(dataPath, x_train_ind, 4)
    with open(dataPath+'id_captionInd_train.pickle', 'wb') as handle:
        pickle.dump(caption_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    caption_data = build_caption_data(dataPath, x_test_ind, 4)
    with open(dataPath+'id_captionInd_test.pickle', 'wb') as handle:
        pickle.dump(caption_data, handle, protocol=pickle.HIGHEST_PROTOCOL)



    # for key,values in captions:
    #     print(key,  values, ":     "," ".join([ind2w[i] for i in values ]))


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
