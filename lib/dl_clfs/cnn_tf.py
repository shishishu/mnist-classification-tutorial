#!/usr/bin/env python
#coding=utf-8
# @file  : cnn_tf
# @time  : 9/22/2019 5:13 PM
# @author: shishishu
# ref to: https://github.com/chiphuyen/stanford-tensorflow-tutorials/blob/master/examples/07_convnet_mnist.py

import sys
sys.path.append('../..')  # to import module at root path
import time
import tensorflow as tf
from lib.dl_clfs.model_tf import ConvNet

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 256, 'number of examples per batch')
tf.app.flags.DEFINE_float('learning_rate', 1e-5, 'learning rate')
# tf.app.flags.DEFINE_float('keep_prob', 0.8, 'keep prob in drop out')
tf.app.flags.DEFINE_integer('num_epoch', 100, 'number of training iterations')
tf.app.flags.DEFINE_integer('skip_epoch', 1, 'print intermediate result per skip epoch')
tf.app.flags.DEFINE_integer('skip_step', 50, 'print intermediate result per skip step')

def main(_):
    cnnParams = {
        'batch_size': FLAGS.batch_size,
        'learning_rate': FLAGS.learning_rate,
        'num_epoch': FLAGS.num_epoch,
        'skip_epoch': FLAGS.skip_epoch,
        'skip_step': FLAGS.skip_step
    }
    convNeter = ConvNet(**cnnParams)
    start = time.time()
    print('Training model: cnn')
    convNeter.build()
    convNeter.run()
    print('time cost is: {:.2f}'.format(time.time() - start))

if __name__ == '__main__':

    tf.app.run()