export MODEL_FLAGS="--image_size 65536 --num_channels 128 --num_res_blocks 3"
export DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule linear"
export TRAIN_FLAGS="--lr 1e-4 --batch_size 1 --save_interval 1000"
python3 scripts/train_audio.py --data_dir ../data/wav/chunks $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
