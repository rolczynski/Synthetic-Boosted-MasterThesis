import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import numpy.polynomial.polynomial as poly
from matplotlib import pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


time = [
    (3, 29),
    (4, 42),
    (5, 74),
    (6, 136),
]

fig, ax1 = plt.subplots(figsize=(5, 3))
for point in time:
    ax1.scatter(*point, color='green', alpha=0.8)

for y in np.arange(20, 140, step=20):
    ax1.axhline(y, xmin=0.0, xmax=1.0, alpha=0.2, color='black', linestyle='--', linewidth=1)

x, y = np.array(time).T
coefs = poly.polyfit(x, y, 4)
new_x = np.linspace(3, 6, num=1000)
new_y = poly.polyval(new_x, coefs)

ax1.plot(new_x, new_y, color='black', alpha=0.4)
ax1.set_xlabel('Model Size (M)')
ax1.set_ylabel('Time (min/epoch)')
ax1.set_xticks([3, 4, 5, 6])
ax1.set_xticklabels(['8', '16', '32', '64'])
# plt.show()
plt.tight_layout()
plt.savefig('images/time.png')
