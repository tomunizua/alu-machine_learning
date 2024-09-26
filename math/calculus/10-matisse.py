#!/usr/bin/env python3
"""Calculates the derivative of a polynomial."""


def poly_derivative(poly):
    """
    Calculates the derivative of a polynomial.
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if len(poly) == 1:
        return [0]

    derivative = [
        coeff * power 
        for power, coeff in enumerate(poly) 
        if power > 0
    ]

    if len(derivative) == 0:
        return [0]

    return derivative
