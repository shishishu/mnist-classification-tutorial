#!/usr/bin/env python
#coding=utf-8
# @file  : lr_clf
# @time  : 9/15/2019 2:49 PM
# @author: shishishu

# import sys
# sys.path.append('../..')  # to import module at root path
import argparse
import time
from ast import literal_eval
from gene_dataset import GeneDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument('--flatten_flag', type=literal_eval, default=False, help='flatten images to one dimension')
parser.add_argument('--expand_flag', type=literal_eval, default=True, help='expand labels to two dimensions')


class LogRegCLF:

    def __init__(self, flatten_flag, expand_flag):
        self.data_tr = GeneDataset.load_digits('tr', flatten_flag, expand_flag)
        self.data_te = GeneDataset.load_digits('te', flatten_flag, expand_flag)

    def train_part(self):
        clf = LogisticRegression(
            penalty='l2',
            C=1.0,
            random_state=7,
            solver='liblinear',
            max_iter=100,
            multi_class='ovr',
            n_jobs=1
        )
        clf.fit(self.data_tr['images'], self.data_tr['labels'])
        return clf

    def test_part(self):
        start = time.time()
        clf = self.train_part()
        print('LR training is done, time cost is: {:.2f}'.format(time.time() - start))
        te_label_true = self.data_te['labels']
        te_label_pred = clf.predict(self.data_te['images'])
        acc = accuracy_score(te_label_true, te_label_pred)
        print('test accuracy in LR model is: {:.4f}'.format(acc))


if __name__ == '__main__':

    FLAGS, unparsed = parser.parse_known_args()

    logRegClfer = LogRegCLF(FLAGS.flatten_flag, FLAGS.expand_flag)
    logRegClfer.test_part()