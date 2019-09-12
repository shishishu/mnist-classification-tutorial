#!/usr/bin/env python
#coding=utf-8
# @file  : config
# @time  : 9/11/2019 11:33 PM
# @author: shishishu

import os

current_path = os.path.abspath(__file__)
ROOT_PATH = os.path.abspath(os.path.dirname(current_path) + os.path.sep + '..')

DATA_DIR = os.path.join(ROOT_PATH, 'data')
LOG_DIR = os.path.join(ROOT_PATH, 'log')
MODEL_DIR = os.path.join(ROOT_PATH, 'model')

MNIST_URL = 'http://yann.lecun.com/exdb/mnist/'

MNIST_DICT = {
    'train_images': {'file_name': 'train-images-idx3-ubyte.gz', 'file_size': 9912422, 'example_count': 60000},
    'train_labels': {'file_name': 'train-labels-idx1-ubyte.gz', 'file_size': 28881, 'example_count': 60000},
    'test_images': {'file_name': 't10k-images-idx3-ubyte.gz', 'file_size': 1648877, 'example_count': 10000},
    'test_labels': {'file_name': 't10k-labels-idx1-ubyte.gz', 'file_size': 4542, 'example_count': 10000}
}