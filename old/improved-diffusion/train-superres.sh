export MODEL_FLAGS="--large_size 65536 --small_size=16384 --num_channels 128 --num_res_blocks 3"
export DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule linear"
export TRAIN_FLAGS="--lr 1e-4 --batch_size 1"
python3 scripts/super_res_train_audio.py --data_dir ../data/wav/chunks $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
