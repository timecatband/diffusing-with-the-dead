export MODEL_FLAGS="--image_size 65536 --num_channels 128 --num_res_blocks 3"
export DIFFUSION_FLAGS="--diffusion_steps 200 --noise_schedule linear"
export TRAIN_FLAGS="--lr 1e-4 --batch_size 1"
python3 scripts/sample_audio.py --model_path=model.pt $MODEL_FLAGS $DIFFUSION_FLAGS --batch_size=1
