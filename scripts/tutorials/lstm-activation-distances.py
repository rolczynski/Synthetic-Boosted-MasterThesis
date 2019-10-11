import h5py
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


if __name__ == '__main__':
    with pd.HDFStore('models/2019-03-17/07/evaluation-jurisdic.hdf5', mode='r') as store:
        references = store['references']

    with h5py.File('models/2019-03-17/07/evaluation-jurisdic.hdf5', mode='r') as store:
        lstm_vectors = 13
        fig, axs = plt.subplots(figsize=(15, 5), nrows=1, ncols=3,
                                subplot_kw={'xticks': [], 'yticks': []})
        for ax, sample_id in zip(axs, [100, 200, 300]):
            activation_sample = store[f'outputs/{lstm_vectors}/{sample_id}']
            distances = 1 - cosine_similarity(activation_sample)
            ax.imshow(distances, origin='lower', interpolation='lanczos')
        plt.tight_layout()
        fig.savefig(f'doc/chapters/3.4/lstm-activation-distances.png')
