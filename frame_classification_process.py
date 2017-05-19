'''
process data for frame classification 

'''

import numpy as np 
from PIL import Image
import os
from tqdm import * 


class frame_process(object):

    def __init__(self, num_video, frame_idx = 1, size = (224, 224, 3)):

        self.num_video = num_video
        self.frame_idx = frame_idx
        self.size = size

    def process_original(self):

        curr_path = os.getcwd() + '/datasets/frames'
        
        labels = np.load(os.getcwd() + '/datasets/category.npy')
        
        h, w, c = self.size
        data = np.zeros((self.num_video, h, w, c))
        
        for i in tqdm(range(self.num_video)):
            idx = labels[i, 0]
            video_path = os.path.join(curr_path, idx)
            
            if not os.path.exists(video_path):
                continue
                
            frame_path = video_path + '/frame' + str(self.frame_idx) +'.jpg'
            img = Image.open(frame_path)
            img_resized = img.resize((h, w), Image.ANTIALIAS)
            
            img.load()
            frame = np.asarray(img_resized, dtype = np.float32)

            data[i] = frame

        return data 
    
    
    def process_updated(self, Xind, yind):
        
        frame_path = os.getcwd() + '/datasets/frames'
        
        X = []
        h, w, c = self.size
        for video_idx in tqdm(Xind):
            
            video_path = frame_path + '/video' + str(video_idx)
            img = Image.open(video_path + '/frame' + str(self.frame_idx) + '.jpg')
            img_resized = img.resize((h, w), Image.ANTIALIAS)
            frame = np.asarray(img_resized, dtype = np.float32)
            frame = np.expand_dims(frame, axis=0)
            
            # assert frame has 4 dimension
            assert len(frame.shape) == 4
            
            X.append(frame)
        
        return np.concatenate(X, axis = 0)
    
    
    
    def process_updates_frameSeq_stacked(self, Xind, num_frames = 10):
        frame_dir = os.getcwd() + '/datasets/frames'
        
        h, w, c = self.size
        video_frames = np.zeros( ( num_frames, h, w, c) )
    
        X = []
        # read videos frames
        for video_ind in tqdm(Xind):
            path = frame_dir +'/video'+str(video_ind)
            for fi in range(1, num_frames+1):
                frame = Image.open(path+'/frame'+str(fi)+'.jpg')
                frame = frame.resize( (h,w))
                frame = np.asarray( frame, dtype = np.float32 )
                video_frames[fi-1] = frame
           
            X.append( np.expand_dims(video_frames, axis=0))
        X = np.concatenate(X, axis = 0)
        return X
