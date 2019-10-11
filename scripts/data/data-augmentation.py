import pytest
import numpy as np
from matplotlib import pyplot as plt
import os
os.chdir('/mnt/lun1/rolczynski')

import sys
sys.path.append('/mnt/lun1/rolczynski/DeepSpeech-Keras')

from source.audio import FeaturesExtractor
import source.augmentation as augmentation


def plot(features):
    fix, ax = plt.subplots(figsize=(15, 5))
    ax.imshow(features.T)
    plt.show()


def features():
    feature_extractor = FeaturesExtractor(
        winlen=0.025,
        winstep=0.01,
        nfilt=80,
        winfunc='hamming'
    )
    feat = feature_extractor.get_features(
        files=['DeepSpeech-Keras/tests/data/audio/sent000.wav']
    )[0]
    return (feat-feat.mean(axis=0)) / feat.std(axis=0)


features = features()[:500]

# fix, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 6),
#                                subplot_kw={'xticks': [], 'yticks': []},
#                                gridspec_kw={'hspace': 0, 'wspace': 0})
# ax1.imshow(features.T)
# masked = augmentation.mask_features(np.copy(features), F=20, mf=2, T=50, mt=2)
# ax2.imshow(masked.T)
# plt.show()


fix, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 6),
                               subplot_kw={'xticks': [], 'yticks': []},
                               gridspec_kw={'hspace': 0, 'wspace': 0})
masked = augmentation.mask_features(np.copy(features), F=False, mf=False, T=50, mt=2)
ax1.imshow(masked.T)
masked = augmentation.mask_features(np.copy(features), F=False, mf=False, T=20, ratio_t=.2)
ax2.imshow(masked.T)
plt.show()
pass
