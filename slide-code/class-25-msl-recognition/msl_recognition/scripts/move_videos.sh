#!/bin/bash

# Define source and destination directories
SRC_DIR="/mnt/disk1/ye/exp/MSL4Emergency/msl4emergency-ver-1.0/video"
DEST_DIR="/mnt/disk1/ye/exp/msl_recognition/data/videos/"

# Create the destination directory if it doesn't already exist
mkdir -p "$DEST_DIR"

# Enable case-insensitive globbing to catch files like 'Idx20-8.mp4'
shopt -s nocaseglob

echo "Starting to move video files..."

# Find all .mp4 files inside the subdirectories and move them directly to the destination
find "$SRC_DIR" -mindepth 2 -type f -name "*.mp4" -exec mv -t "$DEST_DIR" {} +

echo "Done! All videos have been moved to $DEST_DIR"

# Disable case-insensitive globbing to return terminal behavior back to normal
shopt -u nocaseglob

