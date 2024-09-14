#!/usr/bin/env python3
"""
Function that Slice a numpy.ndarray along specified axes.
"""


def np_slice(matrix, axes={}):
    """
    Slice a numpy.ndarray along specified axes.
    """
    slices = [slice(*axes.get(i, (None,))) for i in range(matrix.ndim)]
    return matrix[tuple(slices)]
