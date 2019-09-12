#!/usr/bin/env python
#coding=utf-8
# @file  : trans_dump_data
# @time  : 9/12/2019 10:44 PM
# @author: shishishu

import pickle
import os
import numpy as np
from lib.preprocess.download_data import parse_idx_data

def parse_images(imgs, dir_path, data_src, flatten_flag=False):
    num_example = imgs.shape[0]
    if flatten_flag:
        file_name = data_src + '_images_flatten.npy'
        imgs = imgs.reshape((num_example, -1))
    else:
        file_name = data_src + '_images.npy'
    file_path = os.path.join(dir_path, file_name)
    pickle.dump(imgs, open(file_path, 'wb'))

def parse_labels(labels, dir_path, data_src, expand_flag=True):
    num_example = labels.shape[0]
    if expand_flag:
        file_name = data_src + '_labels_expand.npy'
        new_labels = np.zeros((num_example, 10))
        new_labels[np.arange(num_example), labels] = 1
    else:
        file_name = data_src + '_labels.npy'
        new_labels = labels
    file_path = os.path.join(dir_path, file_name)
    pickle.dump(new_labels, open(file_path, 'wb'))


if __name__ == '__main__':
    pass