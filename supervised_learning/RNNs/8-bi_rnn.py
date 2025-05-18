#!/usr/bin/env python3
'''
Bidirectional Cell Forward
'''


import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Perform forward propagation for a bidirectional RNN.
    """
    t, m, i = X.shape
    h = h_0.shape[1]

    H_fwd = np.zeros((t, m, h))
    H_bwd = np.zeros((t, m, h))

    h_prev = h_0
    for step in range(t):
        h_prev = bi_cell.forward(h_prev, X[step])
        H_fwd[step] = h_prev

    h_prev = h_t
    for step in reversed(range(t)):
        h_prev = bi_cell.backward(h_prev, X[step])
        H_bwd[step] = h_prev

    H = np.concatenate((H_fwd, H_bwd), axis=-1)

    Y = np.zeros((t, m, bi_cell.Wy.shape[1]))
    for step in range(t):
        Y[step] = bi_cell.output(H[step])

    return H, Y
