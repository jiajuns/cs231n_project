
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
import pytube
import json
from datetime import datetime
from pytube import YouTube
from pprint import pprint
import imageio
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import cv2
import numpy as np

# current path
curr = os.getcwd()

# need to create datasets file
train_path = curr + '/datasets/train_2017/videodatainfo_2017.json'

with open(train_path) as f:
    train = json.load(f)

Y = []

# frame parameters 
output_frames = 60
video_time = 60

# load data
for video_idx, video in enumerate(train['videos']):

    # get video information
    category = video['category']
    idx = video['video_id']
    # save category label
    Y.append(category)
    url = video['url']
    start_time = video['start time']
    end_time = video['end time']

    # normalize videos datasets by unifying the clip time
    end_time = start_time + video_time

    # download videos
    yt = YouTube(url)
    yt.set_filename(idx)

    try:
        vid = yt.get('mp4')
    except:
        vid = yt.get('mp4', '360p')

    d1 = curr + '/datasets/videos'
    if not os.path.exists(d1):
        os.makedirs(d1)
    vid.download(d1)

    # cut video clip
    oname = d1 + '/' + vid.filename + ".mp4"
    d2 = curr + "/datasets/processed"
    if not os.path.exists(d2):
        os.makedirs(d2)
    tname = d2 + "/processed_" + vid.filename + ".mp4"
    ffmpeg_extract_subclip(oname, start_time, end_time, \
        targetname=tname)

    vidcap = cv2.VideoCapture(tname)

    # total counts of frames
    success,image = vidcap.read()
    tot_count = 0
    success = True
    while success:
        success,image = vidcap.read()
        tot_count += 1

    # read again, totally output 20 frames
    output_interval = tot_count // output_frames

    # how many frames per second
    frame_rate = tot_count // video_time

    success,image = vidcap.read()
    count = 0
    while success:
      success,image = vidcap.read()
      count += 1
      if count % output_interval == 0: # per second
        directory = curr+ "/datasets/frames/" + idx
        if not os.path.exists(directory):
            os.makedirs(directory)
        cv2.imwrite(directory + "/frame%d.jpg" % count, image)

    print('Finish processing {0}'.format(idx))

    # # to simply check codes availability
    # if video_idx == :
    #     break

output_y_dir = curr + '/datasets/' + 'category.npy'
np.save(output_y_dir, np.array(Y))
