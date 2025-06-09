#!/usr/bin/env python3
'''
This module implements PCA to reduce
the dimensionality of input data
'''
import numpy as np


def pca(X, ndim):
    """
    This function implements PCA to reduce
    the dimensionality of the input data
    to a specified number of dimensions.
    """
    # Center the data

    avg = np.mean(X, axis=0, keepdims=True)
    A = X - avg
    u, s, v = np.linalg.svd(A)

    W = v.T[:, :ndim]
    T = np.matmul(A, W)

    return (T)
