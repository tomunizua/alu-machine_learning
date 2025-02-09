#!/usr/bin/env python3
'''
Create a layer with L2 regularization
using tensorflow
'''

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    This function creates a tensorflow layer that includes
    L2 regularization. The layer performs matrix multiplication
    followed by an activation function.

    Arguments:
    prev -- tensor output of the previous layer
    n -- number of nodes in the layer to create
    activation -- activation function that should be used on the layer
    lambtha -- L2 regularization parameter
    """
    l2_reg = tf.contrib.layers.l2_regularizer(lambtha)
    weights_initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    layer = tf.layers.Dense(
        n,
        activation=activation,
        name="layer",
        kernel_initializer=weights_initializer,
        kernel_regularizer=l2_reg)
    return (layer(prev))
