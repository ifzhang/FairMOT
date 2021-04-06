from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import os.path as osp
import cv2
import logging
import argparse
import motmetrics as mm
import numpy as np
import torch

from tracker.multitracker import JDETracker
from tracking_utils import visualization as vis
from tracking_utils.log import logger
from tracking_utils.timer import Timer
from tracking_utils.evaluation import Evaluator
#import datasets.dataset.jde as datasets

from tracking_utils.utils import mkdir_if_missing
from opts import opts
import cv2, time

import supervisely_lib as sly
from supervisely_lib.video_annotation.key_id_map import KeyIdMap


def letterbox(img, height=608, width=1088,
              color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
    return img, ratio, dw, dh


workspace_id = 16
project_id = 1501
video_id = 425115
start_fr = 0
end_fr = 10
model_path = '/alex_work/models/fairmot_dla34.pth'
obj_class_name = 'pedestrain'
obj_class = sly.ObjClass(obj_class_name, sly.Rectangle)

api = sly.Api.from_env()


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
opt = opts().init(['mot'])
opt.load_model = model_path
opt.conf_thres = 0.4


def fairmot_find_pedestrains(opt, project_id, video_id, end_frame=None, start_frame=0, frame_rate=30):
    start_time = time.time()
    meta_json = api.project.get_meta(project_id)
    meta = sly.ProjectMeta.from_json(meta_json)
    key_id_map = KeyIdMap()
    ann_info = api.video.annotation.download(video_id)
    ann = sly.VideoAnnotation.from_json(ann_info, meta, key_id_map)

    if end_frame is None:
        end_frame = ann.frames_count - 1
    tracker = JDETracker(opt, frame_rate=frame_rate)
    ids_to_video_object = {}
    new_frames = []
    for curr_frame in range(start_frame, end_frame):
        new_figures = []
        img_np = api.video.frame.download_np(video_id, frame_index=curr_frame)
        img0 = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        img, _, _, _ = letterbox(img0)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        blob = torch.from_numpy(img).cuda().unsqueeze(0)
        online_targets = tracker.update(blob, img0) # TODO to watch frames logging uncomment src/lib/tracker/multitracker.py strings 375-379
        online_tlwhs = []
        online_ids = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
        for idx, tlwh in enumerate(online_tlwhs):
            if online_ids[idx] not in ids_to_video_object.keys():
                ids_to_video_object[online_ids[idx]] = sly.VideoObject(obj_class)

            left, top, w, h = tlwh

            bottom = top + h
            if round(bottom) >= ann.img_size[0]:
                bottom = ann.img_size[0] - 1
            right = left + w
            if round(right) >= ann.img_size[1]:
                right = ann.img_size[1] - 1

            if left >= ann.img_size[1] or top >= ann.img_size[0] or bottom < 0 or right < 0:
                continue

            if left < 0:
                left = 0
            if top < 0:
                top = 0

            geom = sly.Rectangle(top, left, bottom, right)
            figure = sly.VideoFigure(ids_to_video_object[online_ids[idx]], geom, curr_frame)
            new_figures.append(figure)
        frame_exist = ann.frames.get(curr_frame)
        if frame_exist is None:
            new_frame = sly.Frame(curr_frame, new_figures)
        else:
            new_frame = frame_exist.clone(figures=new_figures)

        new_frames.append(new_frame)

    new_frames_collection = sly.FrameCollection(new_frames)
    new_meta = meta.add_obj_class(obj_class)
    new_objects = sly.VideoObjectCollection(ids_to_video_object.values())
    new_ann = ann.clone(objects=new_objects, frames=new_frames_collection)
    api.project.update_meta(project_id, new_meta.to_json())
    api.video.annotation.append(video_id, new_ann)
    print('Programm work {} seconds for {} frames in project {}'.format(time.time()-start_time, end_frame, project_id))

fairmot_find_pedestrains(opt, project_id, video_id, end_fr, start_fr)

