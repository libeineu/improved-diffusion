MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine --use_kl True"
model_dir=./output/CIFAR10_batch32
export OPENAI_LOGDIR=$model_dir
device=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=$device



mpiexec -n 8 python scripts/image_sample.py --model_path $model_dir/model130000.pt $MODEL_FLAGS $DIFFUSION_FLAGS
