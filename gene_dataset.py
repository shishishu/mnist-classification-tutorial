#!/usr/bin/env python
#coding=utf-8
# @file  : gene_dataset.py
# @time  : 9/12/2019 11:09 PM
# @author: shishishu

import argparse
import os
import pickle
import numpy as np
from ast import literal_eval
from conf import config
from lib.preprocess.download_data import download_mnist, parse_idx_data
from lib.preprocess.trans_dump_data import parse_images, parse_labels

parser = argparse.ArgumentParser()
parser.add_argument('--flatten_flag', type=literal_eval, default=False, help='flatten images to one dimension (row * col)')
parser.add_argument('--expand_flag', type=literal_eval, default=True, help='expand labels to two dimensions (one hot)')


class GeneDataset:

    def __init__(self, flatten_flag, expand_flag):
        self.flatten_flag = flatten_flag
        self.expand_flag = expand_flag
        self.mnist_dict = config.MNIST_DICT
        self.raw_dir = os.path.join(config.DATA_DIR, 'raw')
        self.npy_dir = os.path.join(config.DATA_DIR, 'npy')

    def download_save_data(self):
        for key, val in self.mnist_dict.items():
            unzip_file_path = download_mnist(self.raw_dir, val['file_name'], val['file_size'])
            data = parse_idx_data(unzip_file_path, val['example_count'])
            data_src, data_type = GeneDataset.split_file_name(val['file_name'])
            if data_type == 'images':
                parse_images(data, self.npy_dir, data_src, self.flatten_flag)
            if data_type == 'labels':
                parse_labels(data, self.npy_dir, data_src, self.expand_flag)

    @staticmethod
    def split_file_name(file_name):
        data_src_raw, data_type = file_name.split('-')[:2]
        if data_src_raw == 'train':
            data_src = 'tr'
        if data_src_raw == 't10k':
            data_src = 'te'
        return data_src, data_type

    @staticmethod
    def load_digits(data_src, flatten_flag=False, expand_flag=True):
        data = dict()
        data['images'] = GeneDataset.load_npy_images(data_src, flatten_flag)
        data['labels'] = GeneDataset.load_npy_labels(data_src, expand_flag)
        return data

    @staticmethod
    def load_npy_images(data_src, flatten_flag=False):
        if flatten_flag:
            file_name = data_src + '_images_flatten.npy'
        else:
            file_name = data_src + '_images.npy'
        file_path = os.path.join(config.DATA_DIR, 'npy', file_name)
        return pickle.load(open(file_path, 'rb'))

    @staticmethod
    def load_npy_labels(data_src, expand_flag=True):
        if expand_flag:
            file_name = data_src + '_labels_expand.npy'
        else:
            file_name = data_src + '_labels.npy'
        file_path = os.path.join(config.DATA_DIR, 'npy', file_name)
        return pickle.load(open(file_path, 'rb'))

    @staticmethod
    def convert_npy_to_txt(data_src):  # tensorflow input text
        images_npy = GeneDataset.load_npy_images(data_src, flatten_flag=True).astype(np.float32)
        labels_npy = GeneDataset.load_npy_labels(data_src, expand_flag=True).astype(np.float32)
        row_npy = np.hstack((images_npy, labels_npy))
        # row_list = row_npy.tolist()
        file_name = data_src + '_images_labels.txt'
        file_path = os.path.join(config.DATA_DIR, 'text2', file_name)
        with open(file_path, 'w', encoding='utf-8') as fw:
            for idx in range(row_npy.shape[0]):
                row = list(map(str, row_npy[idx]))
                fw.writelines(' '.join(row))
                fw.write('\n')
        return


if __name__ == '__main__':

    FLAGS, unparsed = parser.parse_known_args()

    # genDater = GeneDataset(FLAGS.flatten_flag, FLAGS.expand_flag)
    # genDater.download_save_data()
    GeneDataset.convert_npy_to_txt('tr')
    GeneDataset.convert_npy_to_txt('te')
