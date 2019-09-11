#!/usr/bin/env python
#coding=utf-8
# @file  : download_data
# @time  : 9/10/2019 11:53 PM
# @author: shishishu

import os
import requests
import logging
import gzip
from conf import config

def download_url(url):
    try:
        result = requests.get(url)
        if result.status_code != 200:
            logging.error('wrong url or network...')
            return None
        return result.content
    except requests.exceptions.RequestException as e:
        logging.error('error occurs at %s' % e)
        return None

def download_unzip_file(content, file_path):
    if content is None:
        return
    if not os.path.exists(file_path):
        with open(file_path, 'wb') as fw:
            fw.write(content)
        file_path_unzip = file_path.replace('.gz', '')
        gzip_file = gzip.GzipFile(file_path)
        with open(file_path_unzip, 'wb') as fw:
            fw.write(gzip_file.read())
        gzip_file.close()
        os.remove(file_path)  # delete original .gz file
        return file_path_unzip


if __name__ == '__main__':

    file_name = config.MNIST_DATA['train_images']['file_name']
    url = config.MNIST_URL + file_name
    file_path = os.path.join(config.DATA_DIR, 'raw', file_name)
    content = download_url(url)
    download_unzip_file(content, file_path)