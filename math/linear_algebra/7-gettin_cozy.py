#!/usr/bin/env python3
"""
Function that concatenates two matrices along a specific axis.
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specific axis.
    """
    if not mat1 and not mat2:
        return []
    if not mat1:
        return mat2[:] if axis == 0 else None
    if not mat2:
        return mat1[:] if axis == 0 else None

    if axis == 0:
        # Check if both matrices have the same number of columns
        if len(mat1[0]) == len(mat2[0]):
            return mat1 + mat2
        else:
            return None
    elif axis == 1:
        # Check if both matrices have the same number of rows
        if len(mat1) == len(mat2):
            return [row1 + row2 for row1, row2 in zip(mat1, mat2)]
        else:
            return None

    return None
