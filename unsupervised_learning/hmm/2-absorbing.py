#!/usr/bin/env python3
'''This module determines if the markov chain is absorbing '''
import numpy as np


def absorbing(P):
    """Determines if a markov chain is absorbing.

    Args:
        P (numpy.ndarray): Square transition matrix.

    Returns:
        bool: True if absorbing, False otherwise.
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return False
    n, n = P.shape
    if n != P.shape[0]:
        return False
    if not np.all(np.sum(P, axis=1) == 1):
        return False
    if np.any(np.diag(P) == 1):
        return True
    return False
