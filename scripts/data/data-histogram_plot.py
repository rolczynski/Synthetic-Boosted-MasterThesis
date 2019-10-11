import pickle
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def load_session():
    with open('data-histogram.py.session', 'rb') as file:
        return pickle.load(file)


def save_histogram(times, word_counts):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.5))
    sns.distplot(times, ax=ax1, label='Clarin-PL', color='green')
    ax1.set_xlabel(r'Time [s]')
    ax1.set_xlim(right=7)

    ax2.hist(word_counts, bins=np.arange(3.5, 8.5, 1), rwidth=0.9,
             color='green', density=True, alpha=0.8)
    ax2.set_xlabel(r'Words Count')
    ax2.set_xticks([4, 5, 6, 7])
    ax2.set_yticks([.1, .2, .3, .4])
    ax2.set_ylim([0, .45])
    plt.tight_layout()
    # plt.show()
    fig.savefig('images/data-histogram.png')


if __name__ == '__main__':
    times, wc = load_session()
    save_histogram(times, wc)
