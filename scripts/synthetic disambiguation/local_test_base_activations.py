import numpy as np
import pickle
from matplotlib import pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


distances, accs, val_accs = pickle.load(open('results.bin', 'rb'))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
x = np.linspace(0, 50, num=51)
for i, (acc, val_acc, alpha, linestyle) in enumerate(zip(accs, val_accs, [0.7, 0.5, 0.6, 0.7, 0.9], [':', '', '--', '', '-'])):
    if i == 1 or i == 3:
        continue
    ax1.plot(x, [0, *acc], color='black', alpha=alpha, linestyle=linestyle)
    ax1.plot(x, [0, *val_acc], color='green', alpha=alpha, linestyle=linestyle, label=f'Test: LSTM-{i+1}')

ax1.set_ylim(0.5, 1)
ax1.set_xticks([1, 5, 10, 20, 50])
ax1.set_xlabel(r'\bf{Epochs}')
ax1.set_ylabel(r'\bf{Accuracy}')
ax1.legend()

for y in np.arange(0, 1, step=0.1):
    ax1.axhline(y, xmin=0.0, xmax=1.0, alpha=0.2, color='black', linestyle='--', linewidth=1)

x = [1, 2, 3, 4, 5]
ax2.plot(x, distances, color='green', alpha=0.5, linewidth=1)
for distance, LSTM in zip(distances, x):
    ax2.scatter(LSTM, distance, color='green', alpha=0.7)

for y in np.arange(0, 0.1, step=0.01):
    ax2.axhline(y, xmin=0.0, xmax=1.0, alpha=0.2, color='black', linestyle='--', linewidth=1)

ax2.set_xlabel(r'\bf{LSTM Number}')
ax2.set_ylabel(r'\bf{L1 Distance Between Class Mean Vectors}')
plt.subplots_adjust(wspace=5)
ax2.set_xticks(x)
fig.tight_layout()
plt.savefig('synthesized_activations.png')
plt.show()
