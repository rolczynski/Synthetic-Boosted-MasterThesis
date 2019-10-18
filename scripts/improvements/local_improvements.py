import numpy as np
import pickle
import seaborn as sns
import distances
from matplotlib import pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

base_distances, syn_distances = pickle.load(open('results.bin', 'rb'))


def plot(ax, h, title=''):
    labels = ['1', '2', '3', '4', '5', '5+']
    sns.heatmap(h.astype(int), square=True, xticklabels=labels, yticklabels=labels,
                annot=True, ax=ax, linewidths=.1, fmt='d', cmap=cmap, cbar=False, center=0)
    ax.set_xlabel(r'\bf{Char Edit Distance}')
    ax.set_ylabel(r'\bf{Word Edit Distance}')
    ax.set_title(title, pad=15)


cmap = sns.diverging_palette(255, 133, l=60, n=7, center="light")
fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(7, 3))
distances.plot(ax1, syn_distances, title=r'\bf{Synthetic Boosted Model Mistakes}')
h, *_ = np.histogram2d(syn_distances.word, syn_distances.char, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 1e10])
h_base, *_ = np.histogram2d(base_distances.word, base_distances.char, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 1e10])
plot(ax2, h_base-h, title=r'\bf{Corrected Mistakes}')
plt.tight_layout()
plt.savefig('experiments-synthetic-lm-improvements.png')
