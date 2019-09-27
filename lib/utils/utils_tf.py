#!/usr/bin/env python
#coding=utf-8
# @file  : utils_tf
# @time  : 9/22/2019 6:01 PM
# @author: shishishu

import tensorflow as tf

def decode_txt(line, num_class): # decode line by line
    columns = tf.string_split([line], delimiter=' ')  # convert line into tensor
    images = tf.string_to_number(columns.values[0:(-1*num_class)], out_type=tf.float32)
    labels = tf.string_to_number(columns.values[(-1*num_class):], out_type=tf.float32)
    return images, labels