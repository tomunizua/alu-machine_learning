#!/usr/bin/env python3
"""
Module for initializing variables for a Gaussian Mixture Model
"""

import numpy as np


def initialize(X, k):
    """
    Initializes variables for a Gaussian Mixture Model
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None

    n, d = X.shape
    pi = np.full(k, 1/k)
    kmeans = __import__('1-kmeans').kmeans
    m, _ = kmeans(X, k)
    S = np.tile(np.eye(d), (k, 1, 1))

    return pi, m, S
