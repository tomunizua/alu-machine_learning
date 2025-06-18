#!/usr/bin/env python3
'''
This module deals with calculating Markov chain probability.
'''

import numpy as np


def markov_chain(P, s, t=1):
    '''
    This function calculates the probability of a Markov chain being
    in a particular state after a specified number of iterations.
    '''
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return None
    n, n = P.shape
    if n != P.shape[0]:
        return None
    if type(s) is not np.ndarray:
        return None
    if s.shape[0] != 1 or s.shape[1] != n:
        return None
    if type(t) is not int:
        return None
    if np.sum(P, axis=1).all() != 1:
        return None
    if np.sum(s) != 1:
        return None
    if t == 0:
        return s
    for i in range(t):
        s = np.matmul(s, P)
    return s
