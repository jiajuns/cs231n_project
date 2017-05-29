
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