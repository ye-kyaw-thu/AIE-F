#!/bin/bash

# What IS the label for idx20-31.mp4?
python tools/lookup_video_label.py --video idx20-31.mp4

# Export the full video→label mapping to CSV
python tools/lookup_video_label.py --csv results/video_label_map.csv

# Print it in the terminal
python tools/lookup_video_label.py --list | head -50

