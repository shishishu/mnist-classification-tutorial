#!/usr/bin/env python
#coding=utf-8
# @file  : neural_network_tf
# @time  : 9/21/2019 2:09 PM
# @author: shishishu

import time
import hashlib
import tensorflow as tf

def fully_connect_layer(inputs, output_unit, activation='relu', keep_prob=1.0):
    # inputs: [B, I]
    input_unit = inputs.get_shape().as_list()[1]  # convert to int, not a tensor
    md5_time = hashlib.md5()
    md5_time.update(str(time.time()).encode('utf-8'))
    with tf.variable_scope('fc_layer' + md5_time.hexdigest()):  # to change scope name dynamically
        weights = tf.get_variable(
            name='ffnn_weight',
            shape=[input_unit, output_unit],  # [I, H]
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        biases = tf.get_variable(
            name='ffnn_bias',
            shape=[output_unit],
            initializer=tf.random_uniform_initializer(-0.01, 0.01)
        )
        output = tf.matmul(inputs, weights) + biases  # [B, H]
        if activation == 'relu':
            output = tf.nn.relu(output)
        if activation == 'tanh':
            output = tf.nn.tanh(output)
        if activation == 'sigmoid':
            output = tf.nn.sigmoid(output)
    return tf.nn.dropout(output, keep_prob)

def fully_connect_layers(inputs, output_units, activation='relu', keep_prob=1.0):
    tmp_inputs = inputs
    for output_unit in output_units:
        tmp_inputs = fully_connect_layer(tmp_inputs, output_unit, activation, keep_prob)
    return tmp_inputs

def softmax_classifier(inputs, num_class):
    # inputs: [B, H]
    input_unit = inputs.get_shape().as_list()[1]
    with tf.variable_scope('softmax'):
        weights = tf.get_variable(
            name='softmax_w',
            shape=[input_unit, num_class],  # [B, C]
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        biases = tf.get_variable(
            name='softmax_b',
            shape=[num_class],
            initializer=tf.random_uniform_initializer(-0.01, 0.01)
        )
    pred = tf.matmul(inputs, weights) + biases  # [B, C]
    prob = tf.nn.softmax(pred, axis=1)  # [B, C]
    return prob