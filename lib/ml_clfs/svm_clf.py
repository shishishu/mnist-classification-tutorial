#!/usr/bin/env python
#coding=utf-8
# @file  : svm_clf
# @time  : 9/15/2019 5:48 PM
# @author: shishishu

# import sys
# sys.path.append('../..')  # to import module at root path
import time
from gene_dataset import GeneDataset
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


class SVMCLF:

    def __init__(self):
        self.data_tr = GeneDataset.load_digits('tr', flatten_flag=False, expand_flag=False)
        self.data_te = GeneDataset.load_digits('te', flatten_flag=False, expand_flag=False)

    def train_part(self):
        sc = StandardScaler()
        X_train = SVMCLF.count_white_dots(self.data_tr['images'])
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        clf = SVC(
            C=1.0,
            kernel='rbf',
            gamma='auto',
            decision_function_shape='ovr'
        )
        clf.fit(X_train_std, self.data_tr['labels'])
        return sc, clf

    def train_part_flatten(self):
        X_train = self.data_tr['images'].reshape(self.data_tr['images'].shape[0], -1)
        clf = SVC(
            C=1.0,
            kernel='rbf',
            gamma='auto',
            decision_function_shape='ovr'
        )
        clf.fit(X_train, self.data_tr['labels'])
        return clf

    def test_part(self):
        start = time.time()
        sc, clf = self.train_part()
        print('SVM training is done, time cost is: {:.2f}'.format(time.time() - start))
        X_test = SVMCLF.count_white_dots(self.data_te['images'])
        X_test_std = sc.transform(X_test)
        te_label_true = self.data_te['labels']
        te_label_pred = clf.predict(X_test_std)
        acc = accuracy_score(te_label_true, te_label_pred)
        print('test accuracy in SVM model is: {:.4f}'.format(acc))

        start = time.time()
        clf = self.train_part_flatten()
        print('SVM flatten training is done, time cost is: {:.2f}'.format(time.time() - start))
        X_test = self.data_te['images'].reshape(self.data_te['images'].shape[0], -1)
        te_label_pred2 = clf.predict(X_test)
        acc2 = accuracy_score(te_label_true, te_label_pred2)
        print('test accuracy in SVM flatten model is: {:.4f}'.format(acc2))

    @staticmethod
    def count_white_dots(input):
        assert input.ndim == 3, 'wrong dim of input...'
        return input.sum(axis=-1)


if __name__ == '__main__':

    svmClfer = SVMCLF()
    svmClfer.test_part()