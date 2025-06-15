#!/usr/bin/env python3
'''
This module contains the function optimum_k,
that tests for the optimum number of clusters by variance.
'''
import numpy as np


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    '''
    Tests for the optimum number of clusters by variance.
    '''
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(kmin, int) or kmin <= 0 or kmin >= X.shape[0]:
        return None, None
    if kmax is None:
        kmax = X.shape[0]
    if not isinstance(kmax, int) or kmax <= 0 or kmax > X.shape[0]:
        return None, None
    if kmin >= kmax:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    kmeans = __import__('1-kmeans').kmeans
    variance = __import__('2-variance').variance

    results = []
    d_vars = []

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        results.append((C, clss))

        if k == kmin:
            base_variance = variance(X, C)
            d_vars.append(0)
        else:
            current_variance = variance(X, C)
            d_vars.append(base_variance - current_variance)

    return results, d_vars
