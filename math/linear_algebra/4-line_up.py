#!/usr/bin/env python3
"""
Function to add two arrays element-wise.
"""


def add_arrays(arr1, arr2):
    """
    Adds two arrays element-wise and returns a new list with the results.
    If arrays are not the same length, returns None.
    """
    if len(arr1) != len(arr2):
        return None
    else:
        result = []
        for i in range(len(arr1)):
            result.append(arr1[i] + arr2[i])
        return result
