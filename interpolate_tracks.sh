#!bin/bash

source /bd_targaryen/users/omnimotion/anaconda3/bin/activate
conda activate ZeqingTracker

cd /bd_targaryen/users/omnimotion/ZeqingTracker/co-tracker

python interpolate_tracks.py --video-path /bd_targaryen/users/omnimotion/ZeqingTracker/Surgical-Visual-Force-Field/endosurf/data/scared2019/dataset/dataset_1_keyframe_1/data/rgb.mp4 --save-dir /bd_targaryen/users/omnimotion/ZeqingTracker/Surgical-Visual-Force-Field/results/sd1

