'''
process data for frame classification 

'''

import numpy as np 
from PIL import Image
import os
from tqdm import * 


class frame_process(object):

    def __init__(self, num_video, frame_idx, size):

        self.num_video = num_video
        self.frame_idx = frame_idx
        self.size = size

    def process(self):

        curr_path = os.getcwd() + '/datasets/frames'

        h, w, c = self.size

        data = np.zeros((num_video, h, w, c))

        for idx in tqdm(range(num_video)):

            video_path = curr_path + '/video' + str(idx)

            frame_path = video_path + '/frame' + str(self.frame_idx)
            img = Image.open(frame_path)
            img.load()
            frame = np.asarray(img, dtype = np.float32)

            data[idx] = frame

        return data 


