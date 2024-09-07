#!/usr/bin/env python3
"""Question 3."""

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 20000)
r = np.log(0.5)
t1 = 5730
t2 = 1600
y1 = np.exp((r / t1) * x)
y2 = np.exp((r / t2) * x)

plt.plot(x, y1, c="red", linestyle="dashed", label="C-14")
plt.plot(x, y2, c="green", label="Ra-226")

plt.margins(x=0, y=0)

plt.xlabel("Time (years)")
plt.ylabel("Fraction Remaining")
plt.title("Exponential Decay of Radioactive Elements")

plt.show()

