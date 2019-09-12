#!/usr/bin/env python
#coding=utf-8
# @file  : download_data
# @time  : 9/10/2019 11:53 PM
# @author: shishishu
# ref: https://github.com/chiphuyen/stanford-tensorflow-tutorials/blob/master/examples/utils.py

import os
import requests
import gzip
import struct
import numpy as np
from conf import config

def download_url(url):
    try:
        result = requests.get(url)
        if result.status_code != 200:
            print('wrong url or network...')
            return None
        return result.content
    except requests.exceptions.RequestException as e:
        print('requests error occurs at %s' % e)
        return None

def download_mnist(dir_path, file_name, expect_bytes=None):
    file_path = os.path.join(dir_path, file_name)
    url = config.MNIST_URL + file_name
    content = download_url(url)
    if content is None:
        print('fail to download data %s' % file_name)
        return
    if not os.path.exists(file_path):
        with open(file_path, 'wb') as fw:
            fw.write(content)
        file_stat = os.stat(file_path)
        if expect_bytes:
            assert file_stat.st_size == expect_bytes, 'wrong file size downloaded: %s' % file_name
        file_path_unzip = file_path.replace('.gz', '')
        gzip_file = gzip.GzipFile(file_path)
        with open(file_path_unzip, 'wb') as fw:
            fw.write(gzip_file.read())
        gzip_file.close()
        os.remove(file_path)  # delete original .gz file
        return file_path_unzip

def parse_idx_data(file_path, expect_count=None):
    _, file_name = os.path.split(file_path)
    data_type = file_name.split('-')[1]
    if data_type == 'images':
        with open(file_path, 'rb') as fr:
            _, num, rows, cols = struct.unpack('>IIII', fr.read(16))
            if expect_count:
                assert num == expect_count, 'wrong count of examples: %s' % file_name
            imgs = np.fromfile(fr, dtype=np.uint8).reshape(num, rows, cols)
            imgs = imgs.astype(np.float32) / 255.0  # normalize in [0, 1]
            return imgs
    if data_type == 'labels':
        with open(file_path, 'rb') as fr:
            _, num = struct.unpack('>II', fr.read(8))
            if expect_count:
                assert num == expect_count, 'wrong count of examples: %s' % file_name
            labels = np.fromfile(fr, dtype=np.int8)  # int8
            return labels


if __name__ == '__main__':
    pass