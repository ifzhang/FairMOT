# FairMOT
A simple baseline for one-shot multi-object tracking:
![](assets/pipeline.png)
## Abstract
There has been remarkable progress on object detection and re-identification in recent years which are the core components for multi-object  tracking.  However,  little  attention  has  been  focused  on  accomplishing the two tasks in a single network to improve the inference speed.The  initial  attempts  along  this  path  ended  up  with  degraded  results mainly because the re-identification branch is not appropriately learned.In this work, we study the essential reasons behind the failure, and accordingly present a simple baseline to addresses the problems. It remarkably outperforms the state-of-the-arts on the MOT challenge datasets at 30 FPS.We hope this baseline could inspire and help evaluate new ideas in this field.

## Tracking performance
### Results on MOT challenge test set
| Dataset    |  MOTA | IDF1 | IDS | MT | ML | FPS |
|--------------|-----------|--------|-------|----------|----------|--------|
|2DMOT15  | 59.0 | 62.2 |  582 | 45.6% | 11.5% | 30.5 |
|MOT16       | 68.7 | 70.4 | 953 | 39.5% | 19.0% | 25.9 |
|MOT17       | 67.5 | 69.8 | 2868 | 37.7% | 20.8% | 25.9 |
|MOT20       | 58.7 | 63.7 | 6013 | 66.3% | 8.5% | 13.2 |

 All of the results are obtained on the [MOT challenge](https://motchallenge.net) evaluation server under the “private detector” protocol. We rank first among all the trackers on 2DMOT15, MOT17 and the recently released (2020.02.29) MOT20. Note that our IDF1 score remarkably outperforms other one-shot MOT trackers by more than **10 points**. The tracking speed of the entire system can reach up to **30 FPS**.

### Video demos on MOT challenge test set
<img src="assets/MOT15.gif" width="400"/>   <img src="assets/MOT16.gif" width="400"/>
<img src="assets/MOT17.gif" width="400"/>   <img src="assets/MOT20.gif" width="400"/>

## Installation
* Clone this repo, and we'll call the directory that you cloned as ${FAIRMOT_ROOT}
* Install dependencies. We use python 3.7 and pytorch >= 1.2.0
```
conda create -n FairMOT
conda activate FairMOT
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
cd ${FAIRMOT_ROOT}
pip install -r requirements.txt
cd src/lib/models/networks/DCNv2 sh make.sh
```
* We use [DCNv2](https://github.com/CharlesShang/DCNv2) in our backbone network and more details can be found in their repo. 
* In order to run the code for demos, you also need to install [ffmpeg](https://www.ffmpeg.org/).

## Data preparation

We use the same training data as [JDE](https://github.com/Zhongdao/Towards-Realtime-MOT). Please refer to their [DATA ZOO](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md) to download and prepare all the training data including Caltech Pedestrian, CityPersons, CUHK-SYSU, PRW, ETHZ, MOT17 and MOT16. 

[2DMOT15](https://motchallenge.net/data/2D_MOT_2015/) and [MOT20](https://motchallenge.net/data/MOT20/) can be downloaded from the official webpage of MOT challenge. After downloading, you should prepare the data in the following structure:
```
MOT15
   |——————images
   |        └——————train
   |        └——————test
   └——————labels_with_ids
            └——————train(empty)
MOT20
   |——————images
   |        └——————train
   |        └——————test
   └——————labels_with_ids
            └——————train(empty)
```
Then, you can change the seq_root and label_root in src/gen_labels_15.py and src/gen_labels_20.py and run:
```
cd src
python gen_labels_15.py
python gen_labels_20.py
```
to generate the labels of 2DMOT15 and MOT20.

## Pretrained models and baseline model
* **Pretrained models**
DLA-34 COCO pretrained model: [DLA-34 official](https://drive.google.com/file/d/1pl_-ael8wERdUREEnaIfqOV_VF2bEVRT/view)
HRNet-W-32 ImageNet pretrained model: [HRNet-W32 official](https://drive.google.com/drive/folders/1E6j6W7RqGhW1o7UHgiQ9X4g8fVJRU9TX)
After downloading, you should put the pretrained models in the following structure:
```
${FAIRMOT_ROOT}
   └——————models
           └——————ctdet_coco_dla_2x.pth
           └——————hrnet_w32-36af842e.pth
```
* **Baseline model**
Our baseline FairMOT model can be downloaded here: [[Google]](https://drive.google.com/open?id=1udpOPum8fJdoEQm6n0jsIgMMViOMFinu)
After downloading, you should put the pretrained models in the following structure:
```
${FAIRMOT_ROOT}
   └——————models
           └——————all_dla34.pth
           └——————...
```

## Training
* Download the training data
* Change the dataset root directory 'root' in src/lib/cfg/data.json and 'data_dir' in src/lib/opts.py
* Run:
```
sh experiments/all_dla34.sh
```

## Tracking
* The default settings run tracking on the validation dataset from 2DMOT15. You can run:
```
cd src
python track.py mot --load_model ../models/all_dla34.pth --conf_thres 0.6
```
to see the tracking results. You can also set save_images=True in src/track.py to save the visualization results of each frame. 

* To get the txt results of the test set of MOT16 or MOT17, you can run:
```
cd src
python track.py mot --test_mot17 True --load_model ../models/all_dla34.pth --conf_thres 0.4
python track.py mot --test_mot16 True --load_model ../models/all_dla34.pth --conf_thres 0.4
```
and send the txt files to the [MOT challenge](https://motchallenge.net) evaluation server to get the results.

* To get the SOTA results of 2DMOT15 and MOT20, you need to finetune the baseline model on the specific dataset because our training set do not contain them. You can run:
```
sh experiments/ft_mot15_dla34.sh
sh experiments/ft_mot20_dla34.sh
```
and then run the tracking code:
```
cd src
python track.py mot --test_mot15 True --load_model your_mot15_model.pth --conf_thres 0.3
python track.py mot --test_mot20 True --load_model your_mot20_model.pth --conf_thres 0.3
```
Results of the test set all need to be evaluated on the MOT challenge server. You can see the tracking results on the training set by setting --val_motxx True and run the tracking code. We set 'conf_thres' 0.4 for MOT16 and MOT17. We set 'conf_thres' 0.3 for 2DMOT15 and MOT20.

## Demo
You can input a raw video and get the demo video by running src/demo.py and get the mp4 format of the demo video:
```
cd src
python demo.py mot --load_model ../models/all_dla34.pth
```
You can change --input-video and --output-root to get the demos of your own videos.

## Acknowledgement
A large potion of the code is borrowed from [Zhongdao/Towards-Realtime-MOT](https://github.com/Zhongdao/Towards-Realtime-MOT) and [xingyizhou/CenterNet](https://github.com/xingyizhou/CenterNet). Thanks for their wonderful works.
