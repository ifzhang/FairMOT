cd src
python train.py mot --exp_id all_dla34 --gpus 0,1 --batch_size 8 --load_model '../models/ctdet_coco_dla_2x.pth'
cd ..