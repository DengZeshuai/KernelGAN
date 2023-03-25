CUDA_VISIBLE_DEVICES=2 python train.py -i ./training_data/X2/ -o ./results/ --SR

# using kernelGAN to estimate kernel
CUDA_VISIBLE_DEVICES=2 python train.py -i /mnt/cephfs/home/dengzeshuai/data/sr/DIV2KRK/lr_x2/ -o ./results_div2krk/

# using kernelGAN for ZSSR on noise data
CUDA_VISIBLE_DEVICES=1 python train.py -i /mnt/cephfs/home/dengzeshuai/data/sr/DIV2KRK/lr_dn_x2/ -o ./results/ --SR --real

# using the cubic kernel for ZSSR
CUDA_VISIBLE_DEVICES=1 python train.py -i /mnt/cephfs/home/dengzeshuai/data/sr/DIV2KRK/lr_dn_x2/ -o ./results_cubic_ZSSR/ --SR --real --cubic