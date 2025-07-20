#!bin/bash

source /bd_targaryen/users/omnimotion/anaconda3/bin/activate
conda activate ZeqingTracker

cd /bd_targaryen/users/omnimotion/ZeqingTracker/Surgical-Visual-Force-Field
python plot_3Dheatmap_ply.py --method nearest



