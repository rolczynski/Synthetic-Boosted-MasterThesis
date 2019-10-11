import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from python_speech_features import get_filterbanks, hz2mel
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


nfilt, nfft, samplerate, lowfreq, highfreq = 7, 512, 16000, 0, 8000
fb = get_filterbanks(nfilt, nfft, samplerate, lowfreq, highfreq)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
colors = sns.cubehelix_palette(7, start=2, rot=0, dark=0.1, light=.7)

x = np.arange(0, 8001, 1)
y = [hz2mel(i) for i in x]

ax1.scatter(1000, 1000, s=30, color='red', alpha=0.9)
ax1.vlines(1000, ymin=0, ymax=1000, alpha=0.8, color='red', linestyle='--', linewidth=1)
ax1.hlines(1000, xmin=0, xmax=1000, alpha=0.8, color='red', linestyle='--', linewidth=1)

ax1.plot(x, y, color='green', alpha=0.7)
for y in np.arange(0, 3000, step=500):
    ax1.axhline(y, xmin=0.0, xmax=1.0, alpha=0.2, color='black', linestyle='--', linewidth=1)

ax1.set_xticks([1000, 2000, 4000, 8000])
ax1.set_xlim(0)
ax1.set_ylim(0)
ax1.set_xlabel(r'\bf{Frequency (Hz)}')
ax1.set_ylabel(r'\bf{Frequency (mel)}')

for i, f in enumerate(fb):
    ax2.plot(f, color=colors[i])

ax2.set_xticks([32, 64, 128, 256])
ax2.set_xticklabels([1000, 2000, 4000, 8000])
ax2.set_xlabel(r'\bf{Frequency (Hz)}')
ax2.set_ylabel(r'\bf{Amplitude}')
ax2.set_xlim(0)
ax2.set_ylim(0)
plt.subplots_adjust(wspace=10)
plt.tight_layout()
plt.savefig('mel-scale.png')
pass
