MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine --use_kl True"
TRAIN_FLAGS="--lr 1e-4  --microbatch 32 --schedule_sampler loss-second-moment"
model_dir=./output/CIFAR10_batch32
if [ ! -d $model_dir ]; then
        mkdir -p $model_dir
fi
export OPENAI_LOGDIR=$model_dir
device=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=$device
mpiexec -n 8 python scripts/image_train.py --data_dir datasets/cifar_train $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
