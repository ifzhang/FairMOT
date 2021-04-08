import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2

def load_func(fpath):
    print('fpath', fpath)
    tree = ET.parse(fpath)
    root = tree.getroot()

    with open(fpath, 'r') as fid:
        lines = fid.readlines()
    records =[json.loads(line.strip('\n')) for line in lines]
    return records


def gen_labels_uadetrac(data_root, label_root, ann_root):
    '''This is a function that read images (sequences) from data_root and annotation data from ann_root
       The generated label data (given in the path label_root) in txt files are the one used for train.py
       TODO: all gen_labels_*.py do not include the traking id information to generate txt under labels_with_ids. 
       TODO: there is no object type from groundtruth written to labels_with_ids txt. Should has this option in the future
    '''
    label_root.mkdir(parents = True, exist_ok = True)
    
    for sequence_path in ann_root.iterdir():
        print('Processing sequence:'+sequence_path.name)
        tree = ET.parse(sequence_path)
        root = tree.getroot()
        fid = -1
        tid_curr = 0
        for ann_data in root.iter():     
            if ann_data.tag == 'frame':
                fid = int(ann_data.attrib['num']) # ID of frame in this sequence
                num_objects = int(ann_data.attrib['density'])
                image_name = 'img{0:05d}.jpg'.format(fid)
                #print(i)
                #anns_data = load_func(ann_root)
                #image_name = '{}.jpg'.format(ann_data['ID'])
                img_path = data_root/sequence_path.stem/image_name             
                img = cv2.imread(
                    str(img_path),
                    cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
                )
                img_height, img_width = img.shape[0:2]

                tid_curr = 0
                for target in list(ann_data[0]):
                    #tid_curr = int(target.attrib['id'])
                    
                    anns = target[0].attrib
                    # Control target objects by the conditions provided in target[1].attrib
                    # There are {orientation, speed, trajectory_length', 'truncation_ratio', 'vehicle_type'}
                    #if target[1].attrib['vehicle_type'] is not 'car':
                    #    continue
                    x, y, w, h = float(anns['left']), float(anns['top']), float(anns['width']), float(anns['height'])
                    x += w / 2
                    y += h / 2
                    label_fpath = (label_root/sequence_path.stem/image_name).with_suffix('.txt')
                    
                    label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                        tid_curr, x / img_width, y / img_height, w / img_width, h / img_height)
                    
                    if not label_fpath.parents[0].exists():
                        label_fpath.parents[0].mkdir(parents=True)

                    with label_fpath.open('a') as f:
                        f.write(label_str)
                    tid_curr += 1


if __name__ == '__main__':
    root_path = Path('../../MOT_data/UA-DETRAC')

    path_images = root_path/'DETRAC-Images'
    
    label_train = root_path/'labels_with_ids/train'
    ann_train = root_path / 'DETRAC-Train-Annotations-XML'

    label_val = root_path/'labels_with_ids/val'
    ann_val = root_path / 'DETRAC-Test-Annotations-XML'

    gen_labels_uadetrac(path_images, label_train, ann_train)
    gen_labels_uadetrac(path_images, label_val, ann_val)