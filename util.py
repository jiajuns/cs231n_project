
# set up
import numpy as np 
import pickle
import os 


def minibatches(input_frames, captions, batch_size):
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
    # video_ind = [captions[i][0] for i in random_index]
    video_ind = range(100)
    batch_input_frames = input_frames[video_ind]
    batch_input_captions = [captions[i][1] for i in random_index]

    return (batch_input_frames, batch_input_captions)


def build_word_to_index_dict():
    curPath = os.getcwd()
    dataPath = curPath + "/datasets/train_2017/"
    
    w2v = pickle.load(open(dataPath+"word2Vector.pickle", "rb"))
    id_cap_dict = pickle.load(open(dataPath+"id_caption_dict.pickle", "rb"))
    
    w2v_keys = list(w2v.keys())
    w2v_keys_sorted = sorted(w2v_keys)
    
    w2ind_dict = {}
    ind2w_dict = {}
    
    for ind, word in enumerate(w2v_keys_sorted):
        w2ind_dict[word] = ind
        ind2w_dict[ind] = word
    
    # check
    print("the index of bardot:  " + str(w2ind_dict['bardot']))
    print("the word of index 1573:  " + str(ind2w_dict[1573]))
    
    
    # store
    with open(dataPath+'word_to_index.pickle', 'wb') as handle:
        pickle.dump(w2ind_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(dataPath+'index_to_word.pickle', 'wb') as handle:
        pickle.dump(ind2w_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

def build_caption_data():
    curPath = os.getcwd()
    dataPath = curPath + "/datasets/train_2017/"
    
    w2ind = pickle.load(open(dataPath+"word_to_index.pickle", "rb"))
    id_cap_dict = pickle.load(open(dataPath+"id_caption_dict.pickle", "rb"))

    caption_data = []
    for video_id, captionLs in id_cap_dict.items():
        video_id = video_id[5:]
        for caption in captionLs:
            video_caption_pair = tuple([int(video_id), caption])
            caption_data.append(video_caption_pair)
    
    with open(dataPath+'video_caption_pairLs.pickle', 'wb') as handle:
        pickle.dump(caption_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    curPath = os.getcwd()
    dataPath = curPath + "/datasets/train_2017/"
    with open(dataPath+'video_caption_pairLs.pickle', 'rb') as handle:
        captions = pickle.load(handle)

    print('Total number of captions: ', len(captions))
    input_frames = np.random.randn(100, 15, 7, 7, 512)
    batch_f, batch_c = minibatches(input_frames, captions, 64)
    print('batch caption: ', batch_c[0])