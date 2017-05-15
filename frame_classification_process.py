'''
process data for frame classification 

'''

import numpy as np 
from PIL import Image
import os
from tqdm import * 


class frame_process(object):

    def __init__(self, num_video, frame_idx, size = (224, 224, 3)):

        self.num_video = num_video
        self.frame_idx = frame_idx
        self.size = size

    def process(self):

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
            img_resized = img.resize((h, w),Image.ANTIALIAS)
            
            img.load()
            frame = np.asarray(img_resized, dtype = np.float32)

            data[i] = frame

        return data 


