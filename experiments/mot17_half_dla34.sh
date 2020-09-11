cd src
python train.py mot --exp_id mot17_half_dla34 --load_model '../models/ctdet_coco_dla_2x.pth' --data_cfg '../src/lib/cfg/mot17_half.json'
cd ..