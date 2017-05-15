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
import multiprocessing
import shutil

# download plugins
imageio.plugins.ffmpeg.download()
from youtube_dl import YoutubeDL
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import cv2
import numpy as np

import logging

log_dir = os.path.join(os.getcwd(), 'log', 'log.txt')
logging.basicConfig(filename=log_dir, level=logging.INFO)

class Download_Video(object):

    def __init__(self, download_num=10, output_num = 10, video_time = 60, max_video_size=1500000):
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
        self.max_video_size = max_video_size

    @staticmethod
    def downloader(task_info):
        video_id, video_url, file_path = task_info

        # download videos
        yt = YouTube(video_url)
        yt.set_filename(video_id)

        try:
            vid = yt.get('mp4')
        except:
            vid = yt.get('mp4', '360p')
        try:
            vid.download(file_path)
            logging.info('Finish downloading {0}'.format(video_id))
        except:
            logging.debug('fail to download {} with url{}'.format(video_id, video_url))

    def download_organizer(self):
        download_count = 0
        ytdl = YoutubeDL(params={'quiet':True})
        d1 = os.path.join(self.curr, 'datasets/videos')
        if not os.path.exists(d1):
            os.makedirs(d1)

        task_list = []
        logging.info('Searching for downloadable video...')
        for video_idx, video in enumerate(self.train['videos']):
            # get video information
            idx = video['video_id']
            url = video['url']

            file_path = os.path.join(d1, idx+'.mp4')

            # if file already exists skip download
            if not os.path.isfile(file_path):
                try:
                    # check video size before downloading
                    info = ytdl.extract_info(url, download=False)
                except:
                    continue

                if info['formats'][0]['filesize'] is not None:
                    if info['formats'][0]['filesize'] < self.max_video_size:
                        task_list.append((idx, url, file_path))
                        download_count += 1

            if download_count >= self.download_num:
                break

        # distribute task across multiple threads:
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        output = pool.map(Download_Video.downloader, task_list)

    @staticmethod
    def processor(task_info):
        video_info, root_path, video_time, output_frames = task_info

        # get video information
        category = video_info['category']
        idx = video_info['video_id']
        start_time = video_info['start time']

        # normalize videos datasets by unifying the clip time
        end_time = start_time + video_time

        oname = os.path.join(root_path, 'videos', idx+'.mp4')
        tname = os.path.join(root_path, 'processed', 'processed_'+idx+".mp4")

        ffmpeg_extract_subclip(oname, start_time, end_time, targetname=tname)
        vidcap = cv2.VideoCapture(tname)

        tot_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        output_interval = tot_count // output_frames

        # how many frames per second
        frame_rate = tot_count // video_time

        success = True
        count, frame_count = 0, 0
        directory = os.path.join(root_path, 'frames', idx)

        while success:
            success, image = vidcap.read()
            count += 1
            if count % output_interval == 0: # per second
                # check the number of output frames
                if frame_count == output_frames:
                    break
                frame_count += 1
                cv2.imwrite(os.path.join(directory, "frame%d.jpg" % count), image)

        logging.info('Finish processing Video {0}'.format(idx))
        return (idx, category)

    def preprocess_organizer(self):
        # cut video clip
        Y = []

        download_count = 0
        root_path = os.path.join(self.curr, 'datasets')
        process_path = os.path.join(root_path, 'processed')

        if not os.path.exists(process_path):
            os.makedirs(process_path)

        if not os.path.exists(os.path.join(root_path, 'frames')):
            os.makedirs(os.path.join(root_path, 'frames'))

        task_list = []
        for video_idx, video_info in enumerate(self.train['videos']):
            idx = video_info['video_id']
            tname = os.path.join(process_path, 'processed_{}.mp4'.format(idx))
            oname = os.path.join(root_path, 'videos', '{}.mp4'.format(idx))
            directory = os.path.join(root_path, 'frames', idx)

            completely_processed = (os.path.isfile(tname) and os.path.exists(directory))
            raw_data_exist = os.path.isfile(oname)

            if (not completely_processed and raw_data_exist):
                download_count += 1
                task_list.append((video_info, root_path, self.video_time, self.output_frames))
                # refresh all the legacy incomplete files
                if not os.path.exists(directory):
                    os.makedirs(directory)

            if download_count >= self.download_num:
                logging.info('Found {} videos for processing'.format(len(task_list)))
                break

        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        output = pool.map(Download_Video.processor, task_list)

        # save category
        output_y_dir = self.curr + '/datasets/category.npy'
        np.save(output_y_dir, np.array(list(output)))