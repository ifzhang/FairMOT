cd src
python train.py mot --exp_id mix_mot17_half_hrnet18 --arch 'hrnet_18' --data_cfg '../src/lib/cfg/data_half.json' --batch_size 8
cd ..