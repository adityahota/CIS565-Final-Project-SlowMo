#!/usr/bin/env bash

# Taken from https://bhupesh-v.github.io/convert-videos-high-quality-gif-ffmpeg/
# Copyright Bhupesh Varshney
# Utility to convert video files to GIFs using ffmpeg
#
# Usage: convert-to-gif.sh <video-file-path> <fps>
# Example:
# convert-to-gif.sh video.mp4 28


if [[ -z "$1" ]]; then
    echo "No video file specified"
    exit 1
fi

if [[ -z "$2" ]]; then
    echo "No fps specified"
    exit 2
fi

# get everything after last /
video=${1##*/}
# remove everything after .
filename=${video%.*}

echo -e "$(tput bold)Getting video dimensions $(tput sgr0)"
# Get video dimensions
dimensions=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 "$1")

echo -e "$(tput bold)Generating Palette $(tput sgr0)"
# Generate palette
ffmpeg -i "$1" -vf "fps=$2,scale=${dimensions%x*}:-1:flags=lanczos,palettegen" "$filename".png

echo -e "$(tput bold)Converting Video to GIF $(tput sgr0)"

ffmpeg -i "$1" -i "$filename".png -filter_complex "fps=$2,scale=${dimensions%x*}:-1:flags=lanczos[x];[x][1:v]paletteuse" "$filename".gif

echo -e "Removing palette"
rm "$filename".png
