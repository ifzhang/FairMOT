from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import argparse
import torch
import json
import time
import os
import cv2

import numpy as np
import torch.nn.functional as F
import datasets.dataset.jde as datasets
from torchvision.transforms import transforms as T
from models.model import create_model, load_model
from models.decode import mot_decode
from models.utils import _tranpose_and_gather_feat
from utils.post_process import ctdet_post_process


max_per_image = 500
num_classes = 1
img_size = (640, 640)
gpu = True
reid_dim = 128
arch = 'dla_34'
ltrb = True
reg_offset = True
conf_thres = 0.3
Kt = 500
heads = {'hm': num_classes, 'wh': 2 if not ltrb else 4, 'id': reid_dim, 'reg': 2}
head_conv = 256
down_ratio = 4
loadp = '../weights/fairmot_dla34.pth'
if gpu:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('Creating model...') 
print('arch, heads, head_conv, device',arch, heads, head_conv,device)
model = create_model(arch, heads, head_conv)
model = load_model(model, loadp)
#model = torch.nn.DataParallel(model)
model = model.to(device)
#model.cuda()
model.eval()
    

def post_process(dets, meta):
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    dets = ctdet_post_process(dets.copy(), [meta['c']], [meta['s']],meta['out_height'], meta['out_width'], num_classes)
    for j in range(1, num_classes + 1):
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
    #print('dets',dets[0].keys())
    return dets[0]

def merge_outputs(detections):
    results = {}
    for j in range(1, num_classes + 1):
        results[j] = np.concatenate([detection[j] for detection in detections], axis=0).astype(np.float32)

    scores = np.hstack([results[j][:, 4] for j in range(1, num_classes + 1)])
    if len(scores) > max_per_image:
        kth = len(scores) - max_per_image
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, num_classes + 1):
            keep_inds = (results[j][:, 4] >= thresh)
            results[j] = results[j][keep_inds]
    return results
    

    
# Get dataloader
inp_path = '/home/boson/Downloads/crowd_square/'
dataset = dataloader = datasets.LoadImages(inp_path, img_size)
saveimg = True
savedir = '../output/'
os.makedirs(savedir, exist_ok=True)

for i, (path, img, img0) in enumerate(dataloader):
    person_count = 0
    im_blob = torch.from_numpy(img).cuda().unsqueeze(0)
    #im_blob = torch.from_numpy(img).unsqueeze(0)
    width = img0.shape[1]
    height = img0.shape[0]
    inp_height = im_blob.shape[2]
    inp_width = im_blob.shape[3]
    c = np.array([width / 2., height / 2.], dtype=np.float32)
    s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
    meta = {'c': c, 's': s,'out_height': inp_height // down_ratio,'out_width': inp_width // down_ratio}

    ''' Step 1: Network forward, get detections & embeddings'''
    with torch.no_grad():
        output = model(im_blob)[-1]
        hm = output['hm'].sigmoid_()
        wh = output['wh']
        id_feature = output['id']
        id_feature = F.normalize(id_feature, dim=1)

        reg = output['reg'] if reg_offset else None
        dets, inds = mot_decode(hm, wh, reg=reg, ltrb=ltrb, K=Kt)
        id_feature = _tranpose_and_gather_feat(id_feature, inds)
        id_feature = id_feature.squeeze(0)
        id_feature = id_feature.cpu().numpy()

    dets = post_process(dets, meta)
    dets = merge_outputs([dets])[1]
    remain_inds = dets[:, 4] > conf_thres
    dets = dets[remain_inds]
    id_feature = id_feature[remain_inds]

    # vis
    person_count+=len(dets)
    for i in range(0, dets.shape[0]):
        bbox = dets[i][0:4]
        cv2.rectangle(img0, (bbox[0], bbox[1]),(bbox[2], bbox[3]),(0, 255, 0), 5)
    print(f'Img: {path} ++ Result: {person_count}')
    print('------------')
    if saveimg:
        cv2.imwrite(os.path.join(savedir,path.split('/')[-1]),img0)
        
    cv2.namedWindow('dets',0)
    cv2.imshow('dets', img0)
    k = cv2.waitKey(0)
    if k == ord('q'):
        break

cv2.destroyAllWindows()
