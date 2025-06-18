#!/usr/bin/env python3
'''
This module deals with initial state probabilities
of a regular markov chain
'''

import numpy as np


def regular(P):
    '''
    Calculates probability of a regular markov chain
    '''
    if len(P.shape) != 2 or P.shape[0] != P.shape[1] or P.shape[0] < 1:
        return None

    P = np.linalg.matrix_power(P, 100)
    if np.any(P <= 0):
        return None

    return np.array([P[0]])
