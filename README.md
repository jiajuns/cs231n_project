# cs231n_project
# Get Started
create google cloud instance  
set up python environment for google cloud  
On terminal run:  
[project path]/setup_googlecloud.sh

# Start Virtual Environment
source .env/bin/activate

# load_data.py
attribute:  
download, preprocess  

Usage:  
from load_data import Download_Video  
video_num = 10  
frame_num = 10 # needed frames for each video  
video_time = 60 # how long videos needed  
d = Download_Video(video_num, frame_num, video_time)  

-- waiting for download  
d.download()  
-- preprocess  
d.preprocess()  

# pretrained_model.py  
Attribute:  
vgg_16_pretrained  
load_features  
split_train_test  
train  
predict  
