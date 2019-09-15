#!/usr/bin/env python
#coding=utf-8
# @file  : run_ml_clf
# @time  : 9/15/2019 4:59 PM
# @author: shishishu

from lib.ml_clfs.lr_clf import LogRegCLF

def run(clfs):
    for clf in clfs:
        clf.test_part()


if __name__ == '__main__':

    clfs = []
    logRegClfer = LogRegCLF(flatten_flag=True, expand_flag=False)
    clfs.append(logRegClfer)

    run(clfs)
