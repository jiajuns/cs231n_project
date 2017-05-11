
#!usr/env/python 2.7
'''
Load Data

One minute per video, One frame per minute

Output:
- videos folder: original downloaded videos
- processed folder: processed videos with shorter time
- frames folder: many frames subfolders
- category.npy: each category label
'''

import os
import json
from datetime import datetime
from pytube import YouTube
from pprint import pprint
import imageio
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import cv2
import numpy as np

class Download_Video(object):

    def __init__(self, download_num=10):
        # current path
        self.curr = os.getcwd()
        self.train_path = os.path.join(self.curr, 'datasets/train_2017/videodatainfo_2017.json')
        with open(self.train_path) as f:
            self.train = json.load(f)
            f.close()

        self.download_num = download_num
        # frame parameters
        self.output_frames = 60
        self.video_time = 60


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

            # vid.set_filename(idx)
            file_path = os.path.join(d1, idx)
            # if file already exists skip download
            if not os.path.isfile(file_path):
                vid.download(d1)

            print('Finish downloading {0}'.format(idx))

            download_count += 1
            if download_count >= self.download_num:
                break


    def preprocess(self):
        # cut video clip
        download_count = 0
        for video_idx, video in enumerate(self.train['videos']):

            # get video information
            category = video['category']
            idx = video['video_id']
            # save category label
            # Y.append(category)
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
                    print('Finish processing {0}'.format(idx))
                    break
                print('Finish processing {0}'.format(idx))
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
            # read again, totally output 20 frames
            output_interval = tot_count // self.output_frames

            # how many frames per second
            frame_rate = tot_count // self.video_time

            vidcap = cv2.VideoCapture(tname)
            success,image = vidcap.read()
            count = 0
            while success:
                success,image = vidcap.read()
                count += 1
                if count % output_interval == 0: # per second
                    directory = os.path.join(self.curr, 'datasets/frames', idx)
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    cv2.imwrite(os.path.join(directory, "frame%d.jpg" % count), image)

            print('Finish processing {0}'.format(idx))


            download_count += 1
            if download_count >= self.download_num:
                break

            # # to simply check codes availability
            # if video_idx == :
            #     break

        # output_y_dir = curr + '/datasets/' + 'category.npy'
        # np.save(output_y_dir, np.array(Y))
