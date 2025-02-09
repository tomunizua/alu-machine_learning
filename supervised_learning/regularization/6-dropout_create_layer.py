#!/usr/bin/env python3
'''
This script creates a dropout layer
using tensorflow library.
'''

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    '''
    The function dropout_create_layer creates a dropout layer.
    It uses the tensorflow library.

    Arguments:
    prev -- tensor output of the previous layer
    n -- number of nodes in the layer to create
    activation -- activation function that should be used on the layer
    keep_prob -- probability that a node will be kept
    '''
    kernel_ini = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    kernel_reg = tf.layers.Dropout(keep_prob)
    layer = tf.layers.Dense(name='layer', units=n, activation=activation,
                            kernel_initializer=kernel_ini,
                            kernel_regularizer=kernel_reg)

    return layer(prev)
