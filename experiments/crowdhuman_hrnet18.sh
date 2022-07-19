cd src
python train.py mot --exp_id crowdhuman_hrnet18 --arch 'hrnet_18'   --num_epochs 60 --lr_step '50' --data_cfg '../src/lib/cfg/crowdhuman.json'
cd ..