cd src
python train.py mot --exp_id all_res50 --gpus 0,1 --batch_size 8 --reid_dim 128 --arch 'resdcn_50'
cd ..