cd src
python train.py mot --exp_id all_hrnet --gpus 0,1 --batch_size 8 --reid_dim 128 --arch 'hrnet_32'
cd ..