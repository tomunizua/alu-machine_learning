#!/usr/bin/env python3
"""Question 4"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

plt.hist(student_grades, bins=range(0, 101, 10), edgecolor="black")

plt.margins(x=0, y=0)

plt.xlabel("Grades")
plt.ylabel("Number of Students")
plt.title("Project A")

plt.xticks(range(0, 101, 10))
plt.ylim(0, 30)

plt.show()

