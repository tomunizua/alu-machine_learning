#!/usr/bin/env python3
"""
Function that Concatenate two numpy.ndarrays along a specified axis.
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Concatenate two numpy.ndarrays along a specified axis.
    """
    return np.concatenate((mat1, mat2), axis=axis)
