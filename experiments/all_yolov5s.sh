cd src
python train.py mot --exp_id all_yolov5s --data_cfg '../src/lib/cfg/data_all.json' --lr 5e-4 --batch_size 16 --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo' --reid_dim 64
cd ..