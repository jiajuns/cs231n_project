
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
import cv2
import numpy as np

class Download_Video(object):

    def __init__(self, download_num=10, output_num = 10, video_time = 60):
        # current path
        self.curr = os.getcwd()
        self.train_path = os.path.join(self.curr, 'datasets/train_2017/videodatainfo_2017.json')
        with open(self.train_path) as f:
            self.train = json.load(f)
            f.close()

        self.download_num = download_num

        # frame parameters
        self.output_frames = output_num
        self.video_time = video_time


    def download(self):
        download_count = 0
        for video_idx, video in enumerate(self.train['videos']):
            # get video information
            idx = video['video_id']
            url = video['url']

            # download videos
            yt = YouTube(url)
            yt.set_filename(idx)

            try:
                vid = yt.get('mp4')
            except:
                vid = yt.get('mp4', '360p')

            d1 = os.path.join(self.curr, 'datasets/videos')
            if not os.path.exists(d1):
                os.makedirs(d1)

            file_path = os.path.join(d1, idx)
            # if file already exists skip download
            if not os.path.isfile(file_path):
                vid.download(d1)
            print('*'*30)
            print('Finish downloading {0}'.format(idx))
            print('*'*30)

            download_count += 1
            if download_count >= self.download_num:
                break


    def preprocess(self):
        # cut video clip
        Y = []
        download_count = 0
        for video_idx, video in enumerate(self.train['videos']):

            # get video information
            category = video['category']
            idx = video['video_id']

            # save category label
            Y.append(category)

            url = video['url']
            start_time = video['start time']
            end_time = video['end time']

            # normalize videos datasets by unifying the clip time
            end_time = start_time + self.video_time

            d1 = os.path.join(self.curr, 'datasets/videos')
            oname = os.path.join(d1, idx+'.mp4')
            d2 = os.path.join(self.curr, 'datasets/processed')

            if not os.path.exists(d2):
                os.makedirs(d2)

            tname = os.path.join(d2, 'processed_'+idx+".mp4")

            if os.path.isfile(tname):
                download_count += 1
                if download_count >= self.download_num:
                    print('Finish preprocess Video {0}'.format(idx))
                    break
                print('Finish preprocess Video {0}'.format(idx))
                continue

            ffmpeg_extract_subclip(oname, start_time, end_time, targetname=tname)
            vidcap = cv2.VideoCapture(tname)

            # total counts of frames
            success, image = vidcap.read()
            tot_count = 0
            success = True
            while success:
                success, image = vidcap.read()
                tot_count += 1

            output_interval = tot_count // self.output_frames

            # how many frames per second
            frame_rate = tot_count // self.video_time

            vidcap = cv2.VideoCapture(tname)
            success,image = vidcap.read()
            count = 0
            frame_count = 0
            while success:
                success,image = vidcap.read()
                count += 1
                if count % output_interval == 0: # per second

                    # check the number of output frames
                    if frame_count == self.output_frames:
                        break
                    frame_count += 1
                    directory = os.path.join(self.curr, 'datasets/frames', idx)
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    cv2.imwrite(os.path.join(directory, "frame%d.jpg" % count), image)

            print('*'*30)
            print('Finish processing Video {0}'.format(idx))
            print('*'*30)

            download_count += 1
            if download_count >= self.download_num:
                break

        # save category
        output_y_dir = self.curr + '/datasets/category.npy'
        np.save(output_y_dir, np.array(Y))
