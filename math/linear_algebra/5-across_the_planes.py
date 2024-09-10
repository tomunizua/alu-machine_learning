#!/usr/bin/env python3
"""
Function that adds two matrices element-wise.
"""


def add_matrices2D(mat1, mat2):
    """
    Adds two 2D matrices element-wise and returns a new matrix.
    """
    if len(mat1) != len(mat2):
        return None
    for row1, row2 in zip(mat1, mat2):
        if len(row1) != len(row2):
            return None
    result = []
    for row1, row2 in zip(mat1, mat2):
        row_result = [a + b for a, b in zip(row1, row2)]
        result.append(row_result)
    return result
