#!/usr/bin/env python3
'''
Implement the L2 regularization cost function
'''
import tensorflow as tf


def l2_reg_cost(cost):
    '''
    Calculate the L2 regularization cost

    Arguments:
    cost -- cost of the neural network without regularization
    '''
    l2_reg_cost = tf.losses.get_regularization_losses()
    return (cost + l2_reg_cost)
