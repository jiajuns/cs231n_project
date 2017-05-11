# cs231n_project

# load_data.py
attribute: \\
download, preprocess \\
from load_data import Download_Video \\
video_num = 10 \\
frame_num = 10 # needed frames for each video \\
video_time = 60 # how long videos needed \\
d = Download_Video(video_num, frame_num, video_time) \\

-- waiting for download \\
d.download() \\
-- preprocess \\
d.preprocess() \\

# pretrained_model.py \\
attribute: \\
vgg_16_pretrained \\
load_features \\
split_train_test \\
train \\
predict \\
