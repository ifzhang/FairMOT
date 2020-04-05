cd src
python train.py mot --exp_id ft_mot15_dla34 --gpus 0,1 --batch_size 8 --load_model '../models/all_dla34.pth' --data_cfg '../src/lib/cfg/mot15.json' --num_epochs 10 --lr 1e-5
cd ..