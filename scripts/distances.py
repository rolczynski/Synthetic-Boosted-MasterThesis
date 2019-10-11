import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')


def calculate(evaluation: pd.DataFrame):
    incorrect = evaluation.wer != 0
    lengths = evaluation[incorrect].transcript.apply(len)
    word_counts = evaluation[incorrect].transcript.str.split().apply(len)

    word_dist = (evaluation[incorrect].wer * word_counts).astype(int)
    char_dist = (evaluation[incorrect].cer * lengths).astype(int)
    distances = pd.DataFrame({
        'word': word_dist,
        'char': char_dist
    })
    return distances


def show(distances: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 6))
    plot(ax, distances)
    plt.show()


def plot(ax, distances: pd.DataFrame, title=''):
    h, *_ = np.histogram2d(distances.word, distances.char, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 1e10])
    labels = ['1', '2', '3', '4', '5', '5+']

    sns.heatmap(h.astype(int), square=True, xticklabels=labels, yticklabels=labels,
                annot=True, ax=ax, linewidths=.1, fmt='d', cmap="YlGnBu", cbar=False)
    ax.set_xlabel(r'\bf{Char Edit Distance}')
    ax.set_ylabel(r'\bf{Word Edit Distance}')
    ax.set_title(title, pad=15)