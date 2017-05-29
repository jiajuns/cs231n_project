
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
    
def caption_to_ind(caption_split, w2ind_dict, maxLen = 20):
    res = []
    for i, word in enumerate(caption_split):
        wordInd = w2ind_dict.get(word)
        res.append(wordInd)
    
    n = len(res)
    for j in range(n, maxLen, 1):
        res.append(None)
        
    return res
        
def build_caption_data(maxLen = 20):
    curPath = os.getcwd()
    dataPath = curPath + "/datasets/train_2017/"
    w2ind = pickle.load(open(dataPath+"word_to_index.pickle", "rb"))
    id_cap_dict = pickle.load(open(dataPath+"id_caption_dict.pickle", "rb"))

    
    caption_data = []
    for video_id, captionLs in id_cap_dict.items():
        video_id = video_id[5:]
        
        for caption in captionLs:
            caption_split = caption.split()
            
            if len(caption_split) > maxLen: # only take captions within maxLen
                break
            
            captionInd = list(caption_to_ind(caption_split, w2ind, maxLen))
            video_caption_pair = tuple([int(video_id), captionInd])
            caption_data.append(video_caption_pair)
            
    with open(dataPath+'video_caption_pairLs.pickle', 'wb') as handle:
        pickle.dump(caption_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        
def check_caption_data():
    curPath = os.getcwd()
    dataPath = curPath + "/datasets/train_2017/"
    caption_data = pickle.load(open(dataPath+"video_caption_pairLs.pickle", "rb"))
    ind2w = pickle.load(open(dataPath+"index_to_word.pickle", "rb"))
    
    print((caption_data[10]))
    for ind in caption_data[10][1]:
        if ind==None:continue
        print (ind2w[ind])

if __name__ == "__main__":
    #build_word_to_index_dict()
    build_caption_data(20)
    check_caption_data()