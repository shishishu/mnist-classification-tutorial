#!/usr/bin/env python
#coding=utf-8
# @file  : neural_network_tf
# @time  : 9/21/2019 2:09 PM
# @author: shishishu

import tensorflow as tf

def fully_connected_layer(inputs, out_dim, activation='relu', keep_prob=1.0, scope_name='fc'):
    # inputs: [B, I]
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        in_dim = inputs.shape[-1]
        W = tf.get_variable(
            name='weights',
            shape=[in_dim, out_dim],  # [I, O]
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        b = tf.get_variable(
            name='biases',
            shape=[out_dim],
            initializer=tf.random_uniform_initializer(-0.01, 0.01)
        )
        output = tf.matmul(inputs, W) + b  # [B, H]
        if activation == 'relu':
            output = tf.nn.relu(output)
        if activation == 'tanh':
            output = tf.nn.tanh(output)
        if activation == 'sigmoid':
            output = tf.nn.sigmoid(output)
        if activation == 'identity':
            pass
    return tf.nn.dropout(output, keep_prob, name=scope.name)

def fully_connected_layers(inputs, out_units, activation='relu', keep_prob=1.0, scope_name='fc'):
    tmp_inputs = inputs
    for idx, out_dim in enumerate(out_units):
        tmp_inputs = fully_connected_layer(tmp_inputs, out_dim, activation, keep_prob, (scope_name + '_' + str(idx)))
    return tmp_inputs

def softmax_classifier(inputs, num_class, scope_name='softmax'):
    # inputs: [B, H]
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        in_dim = inputs.shape[-1]
        W = tf.get_variable(
            name='softmax_W',
            shape=[in_dim, num_class],  # [B, C]
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        b = tf.get_variable(
            name='softmax_b',
            shape=[num_class],
            initializer=tf.random_uniform_initializer(-0.01, 0.01)
        )
        pred = tf.matmul(inputs, W) + b  # [B, C]
    return tf.nn.softmax(pred, axis=1, name=scope.name)


class ConvOperation:

    @staticmethod
    def conv_relu(inputs, filters, k_size, stride, padding, scope_name):
        in_channels = inputs.shape[-1]
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
            kernel = tf.get_variable(
                name='kernel',
                shape=[k_size, k_size, in_channels, filters],
                initializer=tf.truncated_normal_initializer()
            )
            biases = tf.get_variable(
                name='biases',
                shape=[filters],
                initializer=tf.random_normal_initializer()
            )
            conv = tf.nn.conv2d(
                inputs,
                kernel,
                strides=[1, stride, stride, 1],
                padding=padding
            )
        return tf.nn.relu(conv + biases, name=scope.name)

    @staticmethod
    def pool_layer(inputs, ksize, stride, padding='VALID', scope_name='pool', pool_method='max'):
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
            if pool_method == 'max':
                pool = tf.nn.max_pool(
                    inputs,
                    ksize=[1, ksize, ksize, 1],
                    strides=[1, stride, stride, 1],
                    padding=padding
                )
            if pool_method == 'avg':
                pool = tf.nn.avg_pool(
                    inputs,
                    ksize=[1, ksize, ksize, 1],
                    strides=[1, stride, stride, 1],
                    padding=padding
                )
        return pool