#!/usr/bin/env python
#coding=utf-8
# @file  : file_path_op
# @time  : 9/12/2019 10:04 PM
# @author: shishishu

import os

def safe_mkdir(dir_path):
    try:
        os.mkdir(dir_path)
    except FileExistsError:
        pass

def get_dir_file_name(file_path):
    dir_path, file_name = os.path.split(file_path)
    file_name_stem, file_name_suffix = file_name.split('.')
    return dir_path, file_name_stem, file_name_suffix