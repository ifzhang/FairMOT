cd src
python train.py mot --exp_id mix_ft_ch_dla34 --load_model '../models/crowdhuman_dla34.pth' --data_cfg '../src/lib/cfg/data.json'
cd ..