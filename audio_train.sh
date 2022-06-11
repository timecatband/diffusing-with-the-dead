#!/bin/bash
MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 32 --class_cond=False --save_interval=1000"
PYTHONPATH=$PYTHONPATH:/content/improved-diffusion
python3 /content/diffusing-with-the-dead/improved-diffusion/scripts/train_audio.py --data_dir=/content/data/wav/chunks $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
