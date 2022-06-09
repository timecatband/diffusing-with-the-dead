#!/bin/bash
for i in *.mp3; do ffmpeg -i "$i" -ar 22050 "${i%.*}.wav"; done
mkdir wav
mv *.wav wav
