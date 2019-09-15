#!/usr/bin/env python
#coding=utf-8
# @file  : xgb_clf
# @time  : 9/15/2019 7:28 PM
# @author: shishishu

# import sys
# sys.path.append('../..')  # to import module at root path
import multiprocessing
import time
import xgboost as xgb
from gene_dataset import GeneDataset
from lib.ml_clfs.svm_clf import SVMCLF
from sklearn.metrics import accuracy_score


class XGBCLF:

    def __init__(self):
        self.data_tr = GeneDataset.load_digits('tr', flatten_flag=False, expand_flag=False)
        self.data_te = GeneDataset.load_digits('te', flatten_flag=False, expand_flag=False)

    def train_part(self):
        clf = xgb.XGBClassifier(
            max_depth=5,
            min_child_weight=1.0,
            n_jobs=multiprocessing.cpu_count() - 2
        )
        X_train = SVMCLF.count_white_dots(self.data_tr['images'])
        clf.fit(X_train, self.data_tr['labels'])
        return clf

    def train_part_flatten(self):
        clf = xgb.XGBClassifier(
            max_depth=5,
            min_child_weight=1.0,
            n_jobs=multiprocessing.cpu_count() - 2
        )
        X_train = self.data_tr['images'].reshape(self.data_tr['images'].shape[0], -1)
        clf.fit(X_train, self.data_tr['labels'])
        return clf

    def test_part(self):
        start = time.time()
        clf = self.train_part()
        print('XGB training is done, time cost is: {:.2f}'.format(time.time() - start))
        X_test = SVMCLF.count_white_dots(self.data_te['images'])
        te_label_true = self.data_te['labels']
        te_label_pred = clf.predict(X_test)
        acc = accuracy_score(te_label_true, te_label_pred)
        print('test accuracy in XGB model is: {:.4f}'.format(acc))

        start = time.time()
        clf = self.train_part_flatten()
        print('XGB flatten training is done, time cost is: {:.2f}'.format(time.time() - start))
        X_test = self.data_te['images'].reshape(self.data_te['images'].shape[0], -1)
        te_label_pred2 = clf.predict(X_test)
        acc2 = accuracy_score(te_label_true, te_label_pred2)
        print('test accuracy in XGB flatten model is: {:.4f}'.format(acc2))


if __name__ == '__main__':

    xgbClfer = XGBCLF()
    xgbClfer.test_part()