import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import shutil

def load_func(fpath):
    print('fpath', fpath)
    tree = ET.parse(fpath)
    root = tree.getroot()

    with open(fpath, 'r') as fid:
        lines = fid.readlines()
    records =[json.loads(line.strip('\n')) for line in lines]
    return records

def gen_txt_sets_from_anno(dataset_name, data_root, ann_root, train_set = True):
    '''This is a function that read the list of given data under src/data
        The name of the file will be [dataset_name.train] if 'train_set' is True, otherwise it will be [dataset_name.val]
    '''
    if train_set:
        output_path = Path('data')/(dataset_name+'.train')
        # create sub-directories to save images
        if not (data_root/'train').exists():
            (data_root/'train').mkdir()
    else:
        output_path = Path('data')/(dataset_name+'.val')
        # create sub-directories to save images
        if not (data_root/'test').exists():
            (data_root/'test').mkdir()

    

    with output_path.open('w') as f:
        for sequence_path in ann_root.iterdir():
            images_folder_path = data_root/Path(sequence_path).stem
            if not images_folder_path.exists():
                print('Sequence: '+images_folder_path.stem +'does not exist in data path!')
                continue

            if train_set:
                images_folder_path = shutil.move(str(images_folder_path), str(images_folder_path.parents[0]/'train'))
            else:
                images_folder_path = shutil.move(str(images_folder_path), str(images_folder_path.parents[0]/'test'))

            for image_path in Path(images_folder_path).iterdir():
                f.write(str(image_path)+'\n')

def gen_labels_uadetrac(data_root, label_root, ann_root, names):
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
                    if target[1].attrib['vehicle_type'] not in names:
                        print('Error that target object is not in provided class name list !')
                        break

                    x, y, w, h = float(anns['left']), float(anns['top']), float(anns['width']), float(anns['height'])
                    x += w / 2
                    y += h / 2
                    label_fpath = (label_root/sequence_path.stem/image_name).with_suffix('.txt')
                    
                    # Expected format: class id x_center/img_width y_center/img_height w/img_width h/img_height
                    #label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                    #    tid_curr, x / img_width, y / img_height, w / img_width, h / img_height)
                    
                    # With the class id 
                    label_str = '{:d} {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(names.index(target[1].attrib['vehicle_type']),
                        tid_curr, x / img_width, y / img_height, w / img_width, h / img_height)

                    if not label_fpath.parents[0].exists():
                        label_fpath.parents[0].mkdir(parents=True)

                    with label_fpath.open('a') as f:
                        f.write(label_str)
                    tid_curr += 1


if __name__ == '__main__':
    dataset_name = 'UA-DETRAC'
    root_path = Path('../../MOT_data/UA-DETRAC')

    path_images = root_path/'images'
    
    label_train = root_path/'labels_with_ids/train'
    ann_train = root_path / 'DETRAC-Train-Annotations-XML'

    label_val = root_path/'labels_with_ids/val'
    ann_val = root_path / 'DETRAC-Test-Annotations-XML'

    # New classes name for CVAT label.txt
    names = list(('car', 'bus', 'van','others'))
    cvat_label_path = root_path/'for_cvat'/'labels.txt'
    if not cvat_label_path.parents[0].exists():
        cvat_label_path.parents[0].mkdir(parents=True)

    with cvat_label_path.open('w') as out_txt_file:
        for i in names:
            print(i, file =out_txt_file )

    #gen_labels_uadetrac(path_images, label_train, ann_train, names)
    #gen_labels_uadetrac(path_images, label_val, ann_val,names)
    gen_txt_sets_from_anno(dataset_name, path_images, ann_train, train_set = True)
    gen_txt_sets_from_anno(dataset_name, path_images, ann_val, train_set = False)