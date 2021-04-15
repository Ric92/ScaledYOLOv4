import glob
import os
import pickle
import xml.etree.ElementTree as ET
from os import listdir, getcwd
from os.path import join
from voc_to_yolo import (convert_annotation, getImagesInDir)
from zlib import crc32
import numpy as np

def test_set_check(image, test_ratio):
    path, file = os.path.split(image)
    return crc32(file.encode(encoding='UTF-8')) & 0xffffffff < test_ratio *2**32

def split_train_test_by_id(data, test_ratio):
    in_test_set = [test_set_check(value, test_ratio) for value in data]
    return [[data[i] for i in range(len(data)) if in_test_set[i]], [data[i] for i in range(len(data)) if not in_test_set[i]]] 

def check_annotations(dir_path, images_path, annotations_path):
    image_list = []
    unused_image_list = []
    for image_path in images_path:
        basename = os.path.basename(image_path)
        basename_no_ext = os.path.splitext(basename)[0]
        label_path = annotations_path + '/' + basename_no_ext + '.xml'
        if not os.path.exists(label_path):
            unused_image_list.append(image_path)
            continue
        else:
            image_list.append(image_path)
    return [image_list, unused_image_list]

