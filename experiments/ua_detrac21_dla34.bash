cd src
python train.py mot --exp_id ua-detrac21_dla34 --batch_size 8 --gpus 0 --load_model '../models/fairmot_dla34.pth' --data_cfg 'lib/cfg/ua_detrac.json' --data_dir '../../data/UA-DETRAC'
cd ..