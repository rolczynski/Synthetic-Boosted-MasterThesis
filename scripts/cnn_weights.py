import os
os.chdir('/mnt/lun1/rolczynski')

import sys
sys.path.append('/mnt/lun1/rolczynski/DeepSpeech-Keras')

import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from source.utils import load


def save_dense_plot():
    deepspeech = load('DeepSpeech-Keras/2019-03-17/07')
    cnn_layer = deepspeech.model.get_layer('conv2d_1')
    weights, bias = cnn_layer.get_weights()
    weights = np.squeeze(weights, axis=2)

    with pd.HDFStore('DeepSpeech-Keras/2019-03-17/07/evaluation-jurisdic.hdf5', mode='r') as store:
        references = store['references']

    with h5py.File('DeepSpeech-Keras/2019-03-17/07/evaluation-jurisdic.hdf5', mode='r') as store:
        output_index = 4
        activations = np.concatenate([store[f'outputs/{output_index}/{sample_id}']
                                      for sample_id in tqdm(references.index)])
        variances = pd.Series(activations.var(axis=0)).sort_values(ascending=False)

    indices = pd.concat([variances[:17], variances[-1:]]).index.values
    fig, axs = plt.subplots(figsize=(18, 6), nrows=3, ncols=6,
                            subplot_kw={'xticks': [], 'yticks': []})
    for i, ax in zip(indices, axs.flat):
        ax.imshow(weights[:, :, i], aspect='auto', origin='lower', interpolation='lanczos')
    plt.tight_layout()
    fig.savefig(f'doc/chapters/2/images/weights_cnn_1.png')


def save_cnn_plot():
    deepspeech = load('DeepSpeech-Keras/2019-03-30/03')
    cnn_layer = deepspeech.model.get_layer('conv2d_1')
    weights, bias = cnn_layer.get_weights()
    weights = np.squeeze(weights, axis=2)

    with pd.HDFStore('DeepSpeech-Keras/2019-03-30/03/evaluation-jurisdic.hdf5', mode='r') as store:
        references = store['references']

    with h5py.File('DeepSpeech-Keras/2019-03-30/03/evaluation-jurisdic.hdf5', mode='r') as store:
        output_index = 4
        activations = np.concatenate([store[f'outputs/{output_index}/{sample_id}']
                                      for sample_id in tqdm(references.index)])
        variances = pd.Series(activations.var(axis=0).reshape(-1, 80).sum(axis=0)).sort_values(ascending=False)

    indices = variances[:18].index.values
    fig, axs = plt.subplots(figsize=(18, 6), nrows=3, ncols=6,
                            subplot_kw={'xticks': [], 'yticks': []})
    for i, ax in zip(indices, axs.flat):
        ax.imshow(weights[:, :, i], aspect='auto', origin='lower', interpolation='lanczos')
    plt.tight_layout()
    fig.savefig(f'doc/chapters/3.2/images/weights_cnn_2.png')


if __name__ == '__main__':
    # save_dense_plot()
    save_cnn_plot()
