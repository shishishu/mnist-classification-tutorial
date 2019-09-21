#!/usr/bin/env python
#coding=utf-8
# @file  : mlp_tf
# @time  : 9/21/2019 2:08 PM
# @author: shishishu

import sys
sys.path.append('../..')  # to import module at root path
import time
import tensorflow as tf
import numpy as np
from gene_dataset import GeneDataset
import lib.utils.neural_network_tf as nntf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 512, 'number of examples per batch')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
tf.app.flags.DEFINE_string('activation_func', 'relu', 'activation function')
tf.app.flags.DEFINE_float('keep_prob', 0.8, 'keep prob in drop out')
tf.app.flags.DEFINE_string('hidden_layers', '128,32', 'hidden layers in ffnn')
tf.app.flags.DEFINE_integer('num_epoch', 100, 'number of training iterations')
tf.app.flags.DEFINE_integer('skip_epoch', 10, 'print intermediate result per skip')


class MLP:

    def __init__(self, batch_size, learning_rate, activation_func, keep_prob, hidden_layers, num_epoch, skip_epoch):
        self.data_tr = GeneDataset.load_digits('tr', flatten_flag=True, expand_flag=True)  # y: one hot
        self.data_te = GeneDataset.load_digits('te', flatten_flag=True, expand_flag=True)  # y: one hot
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.activation_func = activation_func
        self.keep_prob = keep_prob
        self.hidden_layers = hidden_layers  # list or tuple
        self.num_epoch = num_epoch
        self.skip_epoch = skip_epoch

    def run(self):
        X_train = self.data_tr['images']
        y_train = self.data_tr['labels']
        X_test = self.data_te['images']
        y_test = self.data_te['labels']

        INPUT_SIZE = X_train.shape[1]
        OUTPUT_SIZE = y_train.shape[1]

        X = tf.placeholder(tf.float32, shape=[None, INPUT_SIZE])
        y = tf.placeholder(tf.float32, shape=[None, OUTPUT_SIZE])

        # feed forward part
        X_ffnn = nntf.feed_forward_nn(X, self.hidden_layers, self.activation_func, self.keep_prob)
        y_prob = nntf.softmax_classifier(X_ffnn, OUTPUT_SIZE)

        # back propagation part
        engin_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_prob)
        loss = tf.reduce_mean(engin_loss)
        train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)

        # eval part
        correct_prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_prob, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # run session
        with tf.Session() as sess:
            # initialize variables
            init = tf.global_variables_initializer()
            sess.run(init)

            for epoch in range(1, self.num_epoch + 1):
                get_next_batch = MLP.get_next_batch(X_train, y_train, self.batch_size)
                try:
                    while True:
                        batch_X, batch_y = next(get_next_batch)
                        sess.run([train_op], feed_dict={X: batch_X, y: batch_y})
                except StopIteration:
                    pass
                train_acc = sess.run(accuracy, feed_dict={X: X_train, y: y_train})
                test_acc = sess.run(accuracy, feed_dict={X: X_test, y: y_test})
                print('training accuracy at epoch {} is {:.4f}'.format(epoch, train_acc))
                if epoch % self.skip_epoch == 0:
                    print('test accuracy at epoch {} is {:.4f}'.format(epoch, test_acc))
                if epoch == self.num_epoch:
                    print('final test accuracy is {:.4f}'.format(test_acc))

    @staticmethod
    def get_next_batch(X, y, batch_size):  # generator
        num_total = X.shape[0]
        # shuffle
        perm = np.arange(num_total)
        np.random.shuffle(perm)
        X_train = X[perm]
        y_train = y[perm]
        # iterate
        num_iter = int(num_total / batch_size)
        for idx in range(num_iter):
            start = idx * batch_size
            end = (1 + idx) * batch_size
            yield X_train[start:end], y_train[start:end]

def main(_):
    mlpParams = {
        'batch_size': FLAGS.batch_size,
        'learning_rate': FLAGS.learning_rate,
        'activation_func': FLAGS.activation_func,
        'keep_prob': FLAGS.keep_prob,
        'hidden_layers': list(map(int, FLAGS.hidden_layers.split(','))),
        'num_epoch': FLAGS.num_epoch,
        'skip_epoch': FLAGS.skip_epoch
    }
    mlper = MLP(**mlpParams)
    start = time.time()
    print('Training model: mlp')
    mlper.run()
    print('time cost is: {:.2f}'.format(time.time() - start))


if __name__ == '__main__':

    tf.app.run()
