#!/usr/bin/env python3
'''This module defines the RNNDecoder class'''

import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    '''Decodes for machine translation using RNN with attention'''

    def __init__(self, vocab, embedding, units, batch):
        '''Initialize the RNNDecoder'''
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_state=True,
            return_sequences=True,
            recurrent_initializer="glorot_uniform",
        )
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        '''Executes the decoder logic'''
        units = s_prev.shape[1]
        attention = SelfAttention(units)
        context, _ = attention(s_prev, hidden_states)
        x = self.embedding(x)
        context = tf.expand_dims(context, 1)
        x = tf.concat([context, x], axis=-1)
        y, s = self.gru(x)
        y = tf.reshape(y, (-1, y.shape[2]))
        y = self.F(y)
        return y, s
