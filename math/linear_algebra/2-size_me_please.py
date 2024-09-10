#!/usr/bin/env python3
"""
Write a function def matrix_shape(matrix): that calculates the shape of a matrix
"""


def matrix_shape(matrix):

    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
