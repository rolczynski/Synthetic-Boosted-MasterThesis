import os
import sys
import logging
from logging import Logger
sys.path.append('/mnt/lun1/rolczynski/DeepSpeech-Keras')

import h5py
import pandas as pd
from tqdm import tqdm
os.chdir('/mnt/lun1/rolczynski')        # Evaluation results set up for Cloudlab1


def maybe_load(fname: str):
    if not os.path.isfile(fname):
        print('Loading from Cloudlab2...')
        dst = os.path.abspath(fname)
        directory = os.path.dirname(dst)
        os.makedirs(directory, exist_ok=True)
        src = dst.replace('lun1', 'lun2')
        os.system(f'scp s15497@10.4.4.11:{src} {dst}')


def read(fname: str) -> pd.DataFrame:
    maybe_load(fname)
    with pd.HDFStore(fname, mode='r') as store:
        return store['references']


def read_activations(fname: str, output_index: int = 1):
    maybe_load(fname)
    references = read(fname)
    with h5py.File(fname, mode='r') as store:
        return [store[f'outputs/{output_index}/{sample_id}'][:]
                for sample_id in tqdm(references.index)]


def create_logger(file_path='', level=20, name='error_analysis') -> Logger:
    """ Create the logger and handlers both console and file. """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formater = logging.Formatter('%(asctime)s [%(levelname)-8s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(formater)
    logger.addHandler(console)       # handle all messages from logger (not set handler level)
    if file_path:
        file_handler = logging.FileHandler(file_path, mode='w')
        file_handler.setFormatter(formater)
        logger.addHandler(file_handler)
    return logger
