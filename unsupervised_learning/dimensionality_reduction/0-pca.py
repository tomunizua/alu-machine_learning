#!/usr/bin/env python3
'''
This module implements PCA to reduce dimensionality
'''
import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset.
    """
    # Center the data
    u, s, v = np.linalg.svd(X)
    y = list(x / np.sum(s) for x in s)

    vrce = np.cumsum(y)
    nd = np.argwhere(vrce >= var)[0, 0]

    W = v.T[:, :(nd + 1)]
    return (W)
