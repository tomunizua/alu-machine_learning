#!/usr/bin/env python3
"""Module for finding the best number of clusters for a GMM using BIC"""

import numpy as np

expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a GMM using the Bayesian
    Information Criterion
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None, None
    if kmax is not None and (not isinstance(kmax, int) or kmax <= 0 or
                             kmax < kmin):
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    n, d = X.shape
    if kmax is None:
        kmax = n

    l = []
    b = []
    results = []

    for k in range(kmin, kmax + 1):
        pi, m, S, g, log_likelihood = expectation_maximization(
            X, k, iterations, tol, verbose)

        if pi is None or m is None or S is None or g is None or \
           log_likelihood is None:
            return None, None, None, None

        l.append(log_likelihood)
        results.append((pi, m, S))
        p = k * (d + 2 * d * d - 1) // 2
        bic = p * np.log(n) - 2 * log_likelihood
        b.append(bic)

    l = np.array(l)
    b = np.array(b)

    best_k_index = np.argmin(b)
    best_k = best_k_index + kmin
    best_result = results[best_k_index]

    return best_k, best_result, l, b
