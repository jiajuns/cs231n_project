import numpy as np
from tqdm import tqdm
import glob
import os
import multiprocessing as mp

# image
from scipy.misc import imread, imsave
from PIL import Image
from moviepy.editor import VideoFileClip
from skimage import img_as_float

# CNN model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

def load_features(num_videos, num_frames, h, w, c, model_name='VGG16'):
    '''
    Concanate video frames over short clip period
    frame_dir: frames folder directory
    num_videos: how many videos needed
    '''
    labels = np.load(os.getcwd() + '/datasets/category.npy')
    curr = os.getcwd()
    video_info_list = []
    for i in range(num_videos):
        video_path = os.path.join(curr, 'datasets', 'processed', 'processed_{}.mp4'.format(labels[i, 0]))
        video_info_list.append((video_path, labels[i, 1], num_frames, h, w))

    p = mp.Pool(mp.cpu_count())
    print('processing videos...')

    processed_frames = []
    for i, frames in enumerate(p.imap(process_video, video_info_list)):
        if i % 100 == 0: print('process {0}/{1}'.format(i, num_videos))
        processed_frames.append(frames)

    ytrain = [frames_info[0] for frames_info in processed_frames]
    frames = [frames_info[1] for frames_info in processed_frames]
    frames = np.concatenate(frames, axis=0)
    frames = frames.reshape((-1, h, w, c))

    if model_name == 'VGG16':
        model = vgg_16_pretrained()
    else:
        pass
    print('batch prediction using {} ...'.format(model_name))
    Xtrain = model.predict(frames, batch_size=64)
    Xtrain = Xtrain.reshape((-1, num_frames, 7, 7, 512))
    return Xtrain, ytrain

def vgg_16_pretrained():
    '''
    VGG 16 pretrained model without last 3 fully-connected
    '''
    # vgg without 3 fc
    model = VGG16(weights='imagenet', include_top=False)
    for layer in model.layers:
        layer.trainable = False
    return model

def process_video(video_info):
    video_path = video_info[0]
    video_cateogory = video_info[1]
    output_frames = video_info[2]
    h = video_info[3]
    w = video_info[4]

    vidcap = VideoFileClip(video_path)
    # tot_count = 0
    # for frame in vidcap.iter_frames():
    #     tot_count += 1
    tot_count = int(vidcap.fps * vidcap.duration)

    output_interval = tot_count // output_frames
    count, frame_count = 0, 0

    video_selected_frames = list()
    for frame in vidcap.iter_frames():
        count += 1
        if count % output_interval == 0: # per second
            # check the number of output frames
            if frame_count == output_frames: break
            frame_count += 1
            img = Image.fromarray(frame, 'RGB')
            processed_img = resize_method(img, h, w)
            video_selected_frames.append(processed_img)
    
    del vidcap
    return (video_cateogory, video_selected_frames)

def resize_method(im, h, w):
    im_resized = im.resize((h, w), Image.ANTIALIAS)
    im_arr = np.transpose(np.array(im_resized), (0,1,2))
    im_arr = np.expand_dims(im_resized, axis=0) # add one dimension as 1
    im_arr.flags.writeable = True
    im_arr = im_arr.astype(np.float32)
    im_arr = preprocess_input(im_arr)
    return im_arr

if __name__ == '__main__':
    # testing purpose
    Xtrain, ytrain = load_features(num_videos=2, num_frames=11, h=224, w=224, c=3)
    print(Xtrain.shape)