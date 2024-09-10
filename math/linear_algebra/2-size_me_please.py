#!/usr/bin/env python3
"""
Write a function that calculates the shape of a matrix
"""


def matrix_shape(matrix):
    """
    Returns:
    A list of integers representing the dimensions of the matrix.
    """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
