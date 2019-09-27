#!/usr/bin/env python
#coding=utf-8
# @file  : file_path_op
# @time  : 9/12/2019 10:04 PM
# @author: shishishu

import os
import tensorflow as tf

def safe_mkdir(dir_path):
    try:
        os.makedirs(dir_path)
    except FileExistsError:
        pass

def get_dir_file_name(file_path):
    dir_path, file_name = os.path.split(file_path)
    file_name_stem, file_name_suffix = file_name.split('.')
    return dir_path, file_name_stem, file_name_suffix

def decode_txt(line, num_class): # decode line by line
    columns = tf.string_split([line], delimiter=' ')  # convert line into tensor
    images = tf.string_to_number(columns.values[0:(-1*num_class)], out_type=tf.float32)
    labels = tf.string_to_number(columns.values[(-1*num_class):], out_type=tf.float32)
    return images, labels