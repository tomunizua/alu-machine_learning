#!/usr/bin/env python3
"""Question 6"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

# Labels and colors
fruit_labels = ['apples', 'bananas', 'oranges', 'peaches']
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
people_labels = ['Farrah', 'Fred', 'Felicia']

# Bar positions
x = np.arange(len(people_labels))

# Plot
fig, ax = plt.subplots()

# Create the stacked bars
bottoms = np.zeros(len(people_labels))  # To stack bars on top of each other

for i, (fruit_row, color) in enumerate(zip(fruit, colors)):
    ax.bar(x, fruit_row, width=0.5, bottom=bottoms, color=color, label=fruit_labels[i])
    bottoms += fruit_row

# Labels
ax.set_ylabel('Quantity of Fruit')
ax.set_title('Number of Fruit per Person')
ax.set_xticks(x)
ax.set_xticklabels(people_labels)
ax.set_yticks(np.arange(0, 81, 10))
ax.set_ylim(0, 80)

# Add legend
ax.legend()

plt.show()

