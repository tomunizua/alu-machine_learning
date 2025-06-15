#!/usr/bin/env python3
'''
Module for calculating intra-cluster variance
'''

import numpy as np


def variance(X, C):
    '''
    Calculates the total intra-cluster variance for a data set.
    '''
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None

    try:
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        min_distances = np.min(distances, axis=1)
        var = np.sum(min_distances ** 2)

        return var
    except Exception:
        return None
