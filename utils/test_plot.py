#!/usr/bin/env python3
"""
Test matplotlib plotting functionality
"""

import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for display
import matplotlib.pyplot as plt
import numpy as np

print("Testing matplotlib plotting...")

# Create a simple test plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2)
plt.title('Test Plot - Sine Wave')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()

print("✅ Plot test completed!")
