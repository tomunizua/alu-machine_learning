#!/usr/bin/env python3
"""
Module to calculate the definiteness of a matrix.
"""
import numpy as np


def definiteness(matrix):
    """
    Calculates the definiteness of a matrix.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if len(matrix.shape) != 2 or \
            matrix.shape[0] != matrix.shape[1] or matrix.size == 0:
        return None

    if not np.allclose(matrix, matrix.T):
        return None

    eigenvalues = np.linalg.eigvals(matrix)

    if all(eigenvalues > 0):
        return "Positive definite"
    elif all(eigenvalues >= 0):
        return "Positive semi-definite"
    elif all(eigenvalues < 0):
        return "Negative definite"
    elif all(eigenvalues <= 0):
        return "Negative semi-definite"
    else:
        return "Indefinite"