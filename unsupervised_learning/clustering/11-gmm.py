#!/usr/bin/env python3
"""Module for calculating a GMM from a dataset"""

import sklearn.mixture


def gmm(X, k):
    """Calculates a GMM from a dataset"""
    if not isinstance(X, list) or len(X) == 0:
        return None, None, None, None, None
    if not all(isinstance(row, list) for row in X):
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None

    try:
        X = sklearn.mixture.GaussianMixture._validate_data(X)
    except:
        return None, None, None, None, None

    if len(X.shape) != 2:
        return None, None, None, None, None

    gmm = sklearn.mixture.GaussianMixture(n_components=k)
    gmm.fit(X)

    pi = gmm.weights_
    m = gmm.means_
    S = gmm.covariances_
    clss = gmm.predict(X)
    bic = gmm.bic(X)

    return pi, m, S, clss, sklearn.mixture.GaussianMixture._validate_data([bic])
