#!/usr/bin/env python3

def summation_i_squared(n):
	if isinstance(n, int) and n > 0:
	   return (n * (n + 1) * (2 * n + 1)) // 6
	return None
