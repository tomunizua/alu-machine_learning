#!/usr/bin/env python3
'''
This module contains the SelfAttention class
'''


import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    '''
    This class defines a self-attention mechanism
    '''
    def __init__(self, units):
        '''
        Class constructor
        '''
        super(SelfAttention, self).__init__()
        self.units = units
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        '''
        Method that calls the layer
        '''
        s_prev_expanded = tf.expand_dims(s_prev, 1)

        e = self.V(tf.nn.tanh(self.W(s_prev_expanded) + self.U(hidden_states)))

        weights = tf.nn.softmax(e, axis=1)

        context = tf.reduce_sum(weights * hidden_states, axis=1)

        return context, weights
