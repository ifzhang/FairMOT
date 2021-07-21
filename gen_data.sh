#!/bin/bash

echo "Generating mot data"

python src/track.py mot --test_mot17 True --load_model models/fairmot_dla34.pth --conf_thres 0.4 --data_dir /app/data --exp_id gen_data

python src/track.py mot --val_mot17 True --load_model models/fairmot_dla34.pth --conf_thres 0.4 --data_dir /app/data --exp_id gen_data

python src/track.py mot --test_mot20 True --load_model models/fairmot_dla34.pth --conf_thres 0.3 --data_dir /app/data --exp_id gen_data

python src/track.py mot --val_mot20 True --load_model models/fairmot_dla34.pth --conf_thres 0.3 --data_dir /app/data --exp_id gen_data

echo "DONE!"

gsutil -m rsync /app/data/results/gen_data gs://kyle-reid-tracking-fix/fairmot_base
