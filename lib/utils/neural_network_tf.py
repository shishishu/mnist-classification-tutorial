#!/usr/bin/env python
#coding=utf-8
# @file  : neural_network_tf
# @time  : 9/21/2019 2:09 PM
# @author: shishishu

import tensorflow as tf

def feed_forward_nn(inputs, hidden_layers, activation='relu', keep_prob=1.0):
    final_output = inputs  # inputs: [B, I]
    for idx, num_hidden in enumerate(hidden_layers):
        num_input = final_output.get_shape().as_list()[1]  # convert to int, not a tensor
        with tf.variable_scope('hidden_' + str(idx)):
            weights = tf.get_variable(
                name='ffnn_weight',
                shape=[num_input, num_hidden],  # [I, H]
                initializer=tf.truncated_normal_initializer(stddev=0.1)
            )
            biases = tf.get_variable(
                name='ffnn_bias',
                shape=[num_hidden],
                initializer=tf.random_uniform_initializer(-0.01, 0.01)
            )
        output = tf.matmul(final_output, weights) + biases  # [B, H]
        if activation == 'relu':
            output = tf.nn.relu(output)
        if activation == 'tanh':
            output = tf.nn.tanh(output)
        if activation == 'sigmoid':
            output = tf.nn.sigmoid(output)
        final_output = tf.nn.dropout(output, keep_prob)
    return final_output

def softmax_classifier(inputs, num_class):
    # inputs: [B, H]
    num_softmax_input = inputs.get_shape().as_list()[1]
    with tf.variable_scope('softmax'):
        weights = tf.get_variable(
            name='softmax_w',
            shape=[num_softmax_input, num_class],  # [B, C]
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        biases = tf.get_variable(
            name='softmax_b',
            shape=[num_class],
            initializer=tf.random_uniform_initializer(-0.01, 0.01)
        )
    pred = tf.matmul(inputs, weights) + biases  # [B, C]
    prob = tf.nn.softmax(pred, axis=1)
    return prob