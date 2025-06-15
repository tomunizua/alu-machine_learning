#!/usr/bin/env python3
"""
Module for calculating the PDF of a Gaussian distribution
"""

import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a Gaussian distribution
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None
    if (X.shape[1] != m.shape[0] or X.shape[1] != S.shape[0] or
            S.shape[0] != S.shape[1]):
        return None

    n, d = X.shape
    try:
        det = np.linalg.det(S)
        inv = np.linalg.inv(S)
        constant = 1 / ((2 * np.pi) ** (d / 2) * np.sqrt(det))
        diff = X - m
        exponent = -0.5 * np.sum(np.matmul(diff, inv) * diff, axis=1)
        P = constant * np.exp(exponent)
        return np.maximum(P, 1e-300)
    except Exception as e:
        return None
