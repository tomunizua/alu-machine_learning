#!/usr/bin/env python3
'''
This module contains the positional_encoding function
'''


import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Calculates the positional encoding
    """
    PE = np.zeros((max_seq_len, dm))
    position = np.arange(max_seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, dm, 2) * -(np.log(10000.0) / dm))
    PE[:, 0::2] = np.sin(position * div_term)
    PE[:, 1::2] = np.cos(position * div_term)

    return PE
