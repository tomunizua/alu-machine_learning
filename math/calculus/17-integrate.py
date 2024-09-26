#!/usr/bin/env python3
"""Integration of a polynomial"""


def poly_integral(poly, C=0):
    """
    Calculates the integral of a polynomial.
    """
    if not isinstance(poly, list) or not isinstance(C, int):
        return None

    if len(poly) == 0:
        return None

    integral = [C]

    for power, coeff in enumerate(poly):
        if coeff == 0:
            integral.append(0)
        else:
            new_coeff = coeff / (power + 1)
            if new_coeff.is_integer():
                integral.append(int(new_coeff))
            else:
                integral.append(new_coeff)

    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
