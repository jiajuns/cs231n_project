#!/usr/bin/env bash

# This is the set-up script for Google Cloud.
# we directly use python 3.5
wget http://repo.continuum.io/archive/Anaconda3-4.0.0-Linux-x86_64.sh
bash Anaconda3-4.0.0-Linux-x86_64.sh
source /.bashrc
conda install pip
sudo apt-get install libsm6 libxrender1 libfontconfig1
pip install pytube
conda install -c https://conda.binstar.org/menpo opencv3
conda install pandas==0.19.2  # revert pandas version in order not to have error when importing keras
pip install tensorflow
pip install keras
pip install moviepy


echo "**************************************************"
echo "*****  End of Google Cloud Set-up Script  ********"
echo "**************************************************"
echo ""
echo "In order to use jupyter notebook, follow the instruction" 
echo "in previous cs231n assignment"
