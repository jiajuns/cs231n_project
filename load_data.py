
#!usr/env/python 3.5
'''
Load Data

set download num
set output num 
set video time 

Output:
- videos folder: original downloaded videos
- processed folder: processed videos with shorter time
- frames folder: many frames subfolders
- category.npy: each video category
'''

import os
import json
from datetime import datetime
from pytube import YouTube
from pprint import pprint
import imageio
# download plugins
imageio.plugins.ffmpeg.download()
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip
import numpy as np
from PIL import Image

class Download_Video(object):
    def __init__(self, download_num=10, output_num = 10, video_time = 60):
        # current path
        self.curr = os.getcwd()
        self.train_path = os.path.join(self.curr, 'datasets/train_2017/videodatainfo_2017.json')
        with open(self.train_path) as f:
            self.train = json.load(f)
            f.close()

        self.download_num = download_num
        self.output_frames = output_num         # frame parameters
        self.video_time = video_time


        
    # when download video, download_count doesn't count invalid url. 
    # if download_num=10, it will exactly count 10 videos.
    def download(self):
        
        download_count = 0
        for video_idx, video in enumerate(self.train['videos']):
            
            # get video information
            idx = video['video_id']
            url = video['url']

            # download videos
            try:
                yt = YouTube(url)
                yt.set_filename(idx)
            except:
                print(idx,'url is invalid')
                continue

            try:
                vid = yt.get('mp4')
            except:
                vid = yt.get('mp4', '360p')

            d1 = os.path.join(self.curr, 'datasets/videos')
            if not os.path.exists(d1):
                os.makedirs(d1)

            file_path = os.path.join(d1, idx+'.mp4')
            
            # if file already exists skip download
            if not os.path.isfile(file_path):
                vid.download(d1)
            
            download_count += 1
            if download_count >= self.download_num:
                break

                
    # cut video clip. if download_num=10, it will exactly preprocess 10 videos.
    def preprocess(self):
  
        Y = []
        process_count = 0
        for video_idx, video in enumerate(self.train['videos']):
            # get video information
            category = video['category']
            idx = video['video_id']
            url = video['url']
            
            try:
                yt = YouTube(url)
            except:
                print(idx,'url is invalid')
                continue
                
            # save category label
            Y.append(category)
            start_time = video['start time']
            end_time = video['end time']

            
            # normalize videos datasets by unifying the clip time
            if end_time-start_time > self.video_time:
                end_time = start_time + self.video_time

                       
            # check and create path
            d1 = os.path.join(self.curr, 'datasets/videos')
            oname = os.path.join(d1, idx+'.mp4')
            d2 = os.path.join(self.curr, 'datasets/processed')
            
            if not os.path.exists(oname):
                print('video{0} does not exist'.format(idx))
                continue
                
            if not os.path.exists(d2):
                os.makedirs(d2)
               
            tname = os.path.join(d2, 'processed_'+idx+".mp4")
                      
            
            # cut frames
            ffmpeg_extract_subclip(oname, start_time, end_time, targetname=tname)
            vidcap = VideoFileClip(tname)
            
            # set frames interval
            tot_count = 0 
            for frame in vidcap.iter_frames(): # count total frames
                tot_count += 1
            output_interval = tot_count // self.output_frames
            frame_rate = tot_count // self.video_time

            # cut frames
            vidcap = VideoFileClip(tname)
            count = 0
            frame_count = 0
            for frame in vidcap.iter_frames():
                count += 1
                if count % output_interval == 0: # per second
                    # check the number of output frames
                    if frame_count == self.output_frames:
                        break
                    frame_count += 1
                    img = Image.fromarray(frame, 'RGB')
                    directory = os.path.join(self.curr, 'datasets/frames', idx)
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    img.save(os.path.join(directory, "frame%d.jpg" % frame_count))
             
            process_count += 1 
            if process_count >= self.download_num:
                print('Finish preprocess all videos')
                break

        # save category
        output_y_dir = self.curr + '/datasets/category.npy'
        np.save(output_y_dir, np.array(Y))
