#!/bin/bash
for i in $1/*.mp3; do ffmpeg -i "$i" -ac 1 -ar 22050 "${i%.*}.wav"; done
mv $1/*.wav $2
