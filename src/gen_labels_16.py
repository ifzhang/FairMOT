import os.path as osp
import os
import numpy as np
from pathlib import Path

def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)


seq_root = Path('../../MOT_data/MOT17/images/train')
label_root = Path('../../MOT_data/MOT17/labels_with_ids/train')
label_root.mkdir(parents=True, exist_ok = True)
seqs = [s for s in seq_root.iterdir()]

tid_curr = 0
tid_last = -1

# clear label txt files before append 
for gt_files in label_root.glob('**/**/*.txt'):
    gt_files.unlink()

for seq in seqs:
    seq_info = open(seq/'seqinfo.ini').read()
    seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
    seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])
    total_frames = int(seq_info[seq_info.find('seqLength=') + 10:seq_info.find('\nimWidth')])
    print('Processing sequence:{}, total frames:{}'.format(seq.name, total_frames))
    gt_txt = seq/'gt'/'gt.txt'
    gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')

    seq_label_root = label_root/seq.stem/'img1'
    seq_label_root.mkdir(parents=True, exist_ok = True)

    for fid, tid, x, y, w, h, mark, label, _ in gt:
        if mark == 0 or not label == 1:
            continue
        fid = int(fid)
        tid = int(tid)
        if not tid == tid_last:
            tid_curr += 1
            tid_last = tid
        x += w / 2
        y += h / 2
        label_fpath = seq_label_root/'{:06d}.txt'.format(fid)
        label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
            tid_curr, x / seq_width, y / seq_height, w / seq_width, h / seq_height)
        with label_fpath.open('a') as f:
            f.write(label_str)
