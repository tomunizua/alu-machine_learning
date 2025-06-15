#!/usr/bin/env python3
"""
Module for calculating the maximization step in the EM algorithm for a GMM
"""

import numpy as np


def maximization(X, g):
    """
    Calculates the maximization step in the EM algorithm for a GMM
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None
    if X.shape[0] != g.shape[1]:
        return None, None, None

    n, d = X.shape
    k = g.shape[0]

    if (np.any(g < 0) or np.any(g > 1) or
            not np.allclose(np.sum(g, axis=0), 1)):
        return None, None, None

    pi = np.sum(g, axis=1) / n
    m = np.dot(g, X) / np.sum(g, axis=1)[:, np.newaxis]
    S = np.zeros((k, d, d))
    for i in range(k):
        diff = X - m[i]
        weighted_diff = g[i][:, np.newaxis] * diff
        S[i] = np.dot(diff.T, weighted_diff) / np.sum(g[i])

    return pi, m, S
