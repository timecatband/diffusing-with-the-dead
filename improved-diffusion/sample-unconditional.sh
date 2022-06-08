export MODEL_FLAGS="--large_size 65536 --small_size=16384 --num_channels 128 --num_res_blocks 3"
export DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule linear"
export TRAIN_FLAGS="--lr 1e-4 --batch_size 1"
python3 scripts/super_res_sample_audio.py --model_path=model.pt $MODEL_FLAGS $DIFFUSION_FLAGS --base_samples=$1 --num_batches=1
