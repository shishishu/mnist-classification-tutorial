#!/usr/bin/env python
#coding=utf-8
# @file  : ml_sklearn
# @time  : 9/16/2019 11:52 PM
# @author: shishishu

import sys
sys.path.append('../..')  # to import module at root path
import time
import multiprocessing
import xgboost as xgb
from gene_dataset import GeneDataset
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# decorator
def fit_print_time(func):
    def warp(self, *args, **kwargs):
        print('Training model: {}'.format(func.__name__))
        start = time.time()
        clf = func(self, *args, **kwargs)
        clf.fit(self.data_tr['images'], self.data_tr['labels'])
        print('time cost is: {:.2f}'.format(time.time() - start))
        return clf
    return warp


class MLSklearnClfs:

    def __init__(self):
        self.data_tr = GeneDataset.load_digits('tr', flatten_flag=True, expand_flag=False)
        self.data_te = GeneDataset.load_digits('te', flatten_flag=True, expand_flag=False)

    @fit_print_time
    def lr_clf(self):
        clf = LogisticRegression(
            penalty='l2',
            C=1.0,
            random_state=7,
            solver='liblinear',
            max_iter=100,
            multi_class='ovr',
            n_jobs=1
        )
        return clf

    @fit_print_time
    def svm_clf(self):
        clf = SVC(
            C=1.0,
            kernel='rbf',
            gamma='auto',
            decision_function_shape='ovr'
        )
        return clf

    @fit_print_time
    def xgb_clf(self):
        clf = xgb.XGBClassifier(
            max_depth=5,
            min_child_weight=1.0,
            n_jobs=multiprocessing.cpu_count() - 2
        )
        return clf

    @fit_print_time
    def mlp_clf(self):
        clf = MLPClassifier(
            hidden_layer_sizes=(128, 32),
            activation='relu',
            solver='adam'
        )
        return clf

    def eval_result(self, clf):
        y_true = self.data_te['labels']
        y_pred = clf.predict(self.data_te['images'])
        acc = accuracy_score(y_true, y_pred)
        print('test accuracy is: {:.4f}'.format(acc))


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