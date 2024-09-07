#!/usr/bin/env python3
"""Question 5"""

import numpy as np
import matplotlib.pyplot as plt

# Create a 3x2 grid of subplots
fig, axs = plt.subplots(3, 2, figsize=(10, 8))

# First plot
y0 = np.arange(0, 10) ** 3
x = np.arange(0, 10)
axs[0, 0].plot(x, y0, c="red")
axs[0, 0].set_xlabel("x", fontsize='x-small')
axs[0, 0].set_ylabel("y0", fontsize='x-small')
axs[0, 0].set_title("Plot 1", fontsize='x-small')

# Second plot
mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180
axs[0, 1].plot(x1, y1, linestyle="none", marker="o", markerfacecolor="magenta")
axs[0, 1].set_xlabel("Height (in)", fontsize='x-small')
axs[0, 1].set_ylabel("Weight (lbs)", fontsize='x-small')
axs[0, 1].set_title("Men's Height vs Weight", fontsize='x-small')

# Third plot
x2 = np.arange(0, 28650)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)
axs[1, 0].plot(x2, y2)
axs[1, 0].set_yscale("log")
axs[1, 0].set_xlabel("Time (years)", fontsize='x-small')
axs[1, 0].set_ylabel("Fraction Remaining", fontsize='x-small')
axs[1, 0].set_title("Exponential Decay of C-14", fontsize='x-small')
axs[1, 0].margins(x=0, y=0)

# Fourth plot
x3 = np.arange(0, 20000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)
axs[1, 1].plot(x3, y31, c="red", linestyle="dashed", label="C-14")
axs[1, 1].plot(x3, y32, c="green", label="Ra-226")
axs[1, 1].set_xlabel("Time (years)", fontsize='x-small')
axs[1, 1].set_ylabel("Fraction Remaining", fontsize='x-small')
axs[1, 1].set_title("Exponential Decay of Radioactive Elements", fontsize='x-small')
axs[1, 1].legend(fontsize='x-small')
axs[1, 1].margins(x=0, y=0)

# Fifth plot
np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)
axs[2, 0].hist(student_grades, bins=range(0, 101, 10), edgecolor="black")
axs[2, 0].set_xlabel("Grades", fontsize='x-small')
axs[2, 0].set_ylabel("Number of Students", fontsize='x-small')
axs[2, 0].set_title("Project A", fontsize='x-small')
axs[2, 0].set_xticks(range(0, 101, 10))
axs[2, 0].set_ylim(0, 30)

# Remove the empty subplot and merge last plot to span two columns
fig.delaxes(axs[2, 1])

# Set figure title and adjust layout
fig.suptitle("All in One", y=0.95, fontsize='x-large')
plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.show()

