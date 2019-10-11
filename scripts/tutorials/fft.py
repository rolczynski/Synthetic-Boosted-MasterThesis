"""Tutorials
https://www.youtube.com/watch?v=spUNpyF58BY
"""
import numpy as np
from matplotlib import pyplot as plt

times = np.linspace(0, 0.025, 1000)
tune_G = 43.65
tune_A = 440
func = lambda x: np.sin(2 * np.pi * x * tune_G) + np.sin(2 * np.pi * x * tune_A)   # Unknown: looking for these freq 2 and 3

freqs = np.linspace(0, 1000, 500)
e = lambda x, f: np.exp(-2 * np.pi * 1j * x * f)
single_spec = np.abs([sum(func(t) * e(t, f) for t in times) / len(times) for f in freqs])
plt.plot(single_spec)
plt.show()
