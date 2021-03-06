#!/usr/bin/env python
#coding=utf-8
# @file  : model_tf
# @time  : 4/19/2020 10:56 PM
# @author: shishishu

import tensorflow as tf
import os
import glob
from conf import config
from lib.utils import toolkit
import lib.dl_clfs.layer_tf as ltf
from lib.dl_clfs.layer_tf import ConvOperation as ConvOp
from sklearn.metrics import accuracy_score


# base class
class DNN:

    def __init__(self, batch_size, learing_rate, num_epoch, skip_epoch, skip_step, *args, **kwargs):
        self.batch_size = batch_size
        self.learning_rate = learing_rate
        self.num_epoch = num_epoch
        self.skip_epoch = skip_epoch
        self.skip_step = skip_step
        self.model_dir = os.path.join(config.MODEL_DIR, self.__class__.__name__)
        toolkit.safe_mkdir(self.model_dir)
        self.log_dir = os.path.join(config.LOG_DIR, self.__class__.__name__)
        toolkit.safe_mkdir(self.log_dir)

    def get_data(self, *args, **kwargs):
        with tf.name_scope('data'):
            tr_files = glob.glob(os.path.join(config.DATA_DIR, 'text2', 'tr_*.txt'))
            te_files = glob.glob(os.path.join(config.DATA_DIR, 'text2', 'te_*.txt'))
            print('tr files are: ', tr_files)
            print('te files are: ', te_files)
            print('start read data...')
            tr_data = tf.data.TextLineDataset(tr_files).map(lambda x: toolkit.decode_txt(x, config.NUM_CLASS)).prefetch(10 * self.batch_size)
            tr_data = tr_data.shuffle(buffer_size=10*self.batch_size).batch(self.batch_size)
            te_data = tf.data.TextLineDataset(te_files).map(lambda x: toolkit.decode_txt(x, config.NUM_CLASS))
            te_data = te_data.batch(self.batch_size)
            print('start iterator...')
            iterator = tf.data.Iterator.from_structure(tr_data.output_types, tr_data.output_shapes)
            batch_images, batch_labels = iterator.get_next()

            self.tr_init = iterator.make_initializer(tr_data, name='tr_init')
            self.te_init = iterator.make_initializer(te_data, name='te_init')
            self.imgs = tf.reshape(batch_images, shape=[-1, config.NUM_ROW, config.NUM_COL, 1])  # [B, H, W, Channel]
            self.labels = tf.reshape(batch_labels, shape=[-1, config.NUM_CLASS])  # [B, Class]

    def inference(self, *args, **kwargs):
        pass

    def loss(self, *args, **kwargs):
        with tf.name_scope('loss'):
            engin_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels, logits=self.logits)  # use logits
            self.loss = tf.reduce_mean(engin_loss)

    def optimize(self, *args, **kwargs):
        pass

    def eval(self, *args, **kwargs):
        pass

    def summary(self, *args, **kwargs):
        pass

    def build(self, *args, **kwargs):
        # build the computation graph
        self.get_data()
        self.inference()
        self.loss()
        self.optimize()
        self.eval()
        self.summary()

    def train_one_epoch(self, *args, **kwargs):
        pass

    def eval_epoch(self, *args, **kwargs):
        pass

    def run(self, *args, **kwargs):
        pass


class ConvNet(DNN):

    def __init__(self, batch_size, learning_rate, num_epoch, skip_epoch, skip_step):
        super(ConvNet, self).__init__(batch_size, learning_rate, num_epoch, skip_epoch, skip_step)
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=None, name='keep_prob')

    def inference(self, *args, **kwargs):
        with tf.name_scope('network'):
            conv1 = ConvOp.conv_relu(
                inputs=self.imgs,
                filters=32,
                k_size=5,
                stride=1,
                padding='SAME',
                scope_name='conv_1'
            )
            pool1 = ConvOp.pool_layer(conv1, 2, 2, 'VALID', 'pool_1')
            conv2 = ConvOp.conv_relu(
                inputs=pool1,
                filters=64,
                k_size=5,
                stride=1,
                padding='SAME',
                scope_name='conv_2'
            )
            pool2 = ConvOp.pool_layer(conv2, 2, 2, 'VALID', 'pool_2')
            feature_dim = pool2.shape[1] * pool2.shape[2] * pool2.shape[3]
            pool2 = tf.reshape(pool2, shape=[-1, feature_dim])
            fc = ltf.fully_connected_layer(pool2, 1024, 'relu', self.keep_prob)
            self.logits = ltf.fully_connected_layer(fc, config.NUM_CLASS, 'identity', self.keep_prob, 'logits')

    def optimize(self):
        with tf.name_scope('opt'):
            self.gstep = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, global_step=self.gstep)

    def eval(self):
        with tf.name_scope('predict'):
            self.y_true = tf.argmax(self.labels, axis=1)
            self.prob = tf.nn.softmax(self.logits, axis=1)
            self.y_pred = tf.argmax(self.prob, axis=1)
            correct_preds = tf.equal(self.y_true, self.y_pred)
            self.accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))

    def summary(self):
        with tf.name_scope('summary'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.histogram('hist loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    def train_one_epoch(self, sess, saver, init, writer, epoch, step):
        print('epoch is {}'.format(epoch))
        sess.run(init)
        self.training = True
        total_loss = 0
        n_batches = 0
        print('start training...')
        try:
            while True:
                _, _loss, _summary = sess.run([self.train_op, self.loss, self.summary_op], feed_dict={self.keep_prob: 0.8})
                writer.add_summary(_summary, global_step=step)
                if (step + 1) % self.skip_step == 0:
                    print('loss at step {}: {:.4f}'.format(step, _loss))
                step += 1
                total_loss += _loss
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        if epoch % (10 * self.skip_epoch) == 0:
            saver.save(sess, os.path.join(self.model_dir, self.__class__.__name__), global_step=(step+1))  # prefix: ConvNet/ConvNet-...
        print('average loss at epoch {}: {:.4f}'.format(epoch, total_loss/n_batches))
        return step

    def eval_epoch(self, sess, init, writer, epoch, step):
        sess.run(init)
        self.training = False
        eval_y_true = []
        eval_y_pred = []
        try:
            while True:
                _y_true, _y_pred, _summary = sess.run([self.y_true, self.y_pred, self.summary_op], feed_dict={self.keep_prob: 1.0})
                writer.add_summary(_summary, global_step=step)
                _y_true = _y_true.tolist()
                _y_pred = _y_pred.tolist()
                eval_y_true.extend(_y_true)
                eval_y_pred.extend(_y_pred)
        except tf.errors.OutOfRangeError:
            pass
        print('test accuracy at epoch {}: {:.4f}'.format(epoch, accuracy_score(eval_y_true, eval_y_pred)))

    def run(self):
        # avoid error: CUDNN_STATUS_INTERNAL_ERROR
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        # sess_config.gpu_options.per_process_gpu_memory_fraction = 0.6
        with tf.Session(config=sess_config) as sess:
            writer = tf.summary.FileWriter(self.log_dir, sess.graph)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            # restore model
            ckpt = tf.train.get_checkpoint_state(self.model_dir, latest_filename='checkpoint')
            print('ckpt is: ', ckpt)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('loading saved model: {}'.format(ckpt.model_checkpoint_path))
            else:
                print('new model is created...')
            step = self.gstep.eval()
            for epoch in range(1, self.num_epoch + 1):
                step = self.train_one_epoch(sess, saver, self.tr_init, writer, epoch, step)
                if epoch % self.skip_epoch == 0:
                    self.eval_epoch(sess, self.te_init, writer, epoch, step)
            writer.close()