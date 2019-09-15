#!/usr/bin/env python
#coding=utf-8
# @file  : run_ml_clf
# @time  : 9/15/2019 4:59 PM
# @author: shishishu

from lib.ml_clfs.lr_clf import LogRegCLF
from lib.ml_clfs.svm_clf import SVMCLF
from lib.ml_clfs.xgb_clf import XGBCLF

def run(clfs):
    for clf in clfs:
        clf.test_part()


if __name__ == '__main__':

    clfs = []
    logRegClfer = LogRegCLF()
    clfs.append(logRegClfer)
    svmClfer = SVMCLF()
    clfs.append(svmClfer)
    xgbClfer = XGBCLF()
    clfs.append(xgbClfer)

    run(clfs)
