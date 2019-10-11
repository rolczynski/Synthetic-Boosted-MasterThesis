import pickle
from typing import Any
import pandas as pd
import os
os.chdir('/mnt/lun1/rolczynski')

import sys
sys.path.append('/mnt/lun1/rolczynski/DeepSpeech-Keras')


def dump_session(data: Any):
    with open(f'{__file__}.session', 'wb') as file:
        pickle.dump(data, file)


def read_times_and_wc(store_path: str) -> tuple:
    with pd.HDFStore(store_path, mode='r') as store:
        references = store['references']
        groups = references[['transcript']].groupby('transcript')
        return groups.apply(len).values.astype('int16')


if __name__ == '__main__':
    counts = read_times_and_wc('datasets/mixed/base/train-normalized.hdf5')
    dump_session(counts)

