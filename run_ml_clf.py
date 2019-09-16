#!/usr/bin/env python
#coding=utf-8
# @file  : run_ml_clf
# @time  : 9/15/2019 4:59 PM
# @author: shishishu

from lib.ml_clfs.ml_sklearn import MLSklearnClfs


if __name__ == '__main__':

    mlSklClfer = MLSklearnClfs()
    lr_clf = mlSklClfer.lr_clf()
    mlSklClfer.eval_result(lr_clf)
    svm_clf = mlSklClfer.svm_clf()
    mlSklClfer.eval_result(svm_clf)
    xgb_clf = mlSklClfer.xgb_clf()
    mlSklClfer.eval_result(xgb_clf)
    mlp_clf = mlSklClfer.mlp_clf()
    mlSklClfer.eval_result(mlp_clf)
