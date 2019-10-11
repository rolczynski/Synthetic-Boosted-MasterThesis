import pickle
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
from matplotlib import colors
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def load_session():
    with open('dataset-repetitions.py.session', 'rb') as file:
        return pickle.load(file)


def make_rgb_transparent(rgb, bg_rgb, alpha):
    return [alpha * c1 + (1 - alpha) * c2
            for (c1, c2) in zip(rgb, bg_rgb)]


def save_histogram(counts):
    fig, ax = plt.subplots(figsize=(6, 3))
    size_unique = int(sum(counts == 1) / sum(counts) * 100)
    size_repeated = 100 - size_unique

    # Colors
    green = colors.colorConverter.to_rgb('green')
    green_light = make_rgb_transparent(green, (1, 1, 1), alpha=0.8)
    gray = make_rgb_transparent((0, 0, 0), (1, 1, 1), alpha=0.3)

    # Calculate bins
    bins = [0.5, *np.linspace(1.5, 100.5, num=11), 1000]
    groups = np.digitize(counts, bins)
    results = defaultdict(int)
    for group, value in zip(groups, counts):
        results[group] += value

    values = np.array(list(results.items()))
    sorted_indices = values[:, 0].argsort()
    data = values[sorted_indices]
    data[4, 1] = data[4:, 1].sum()
    data = data[:5, :]
    x, y = data.T

    # Plot
    bars = ax.bar(x, y, color=gray)
    ax.set_xticks([1.5, 2.5, 3.5, 4.5])
    ax.set_xticklabels(['2', '10', '20', '30'])
    ax.set_yticks([5e4, 1e5, 1.5e5, 2e5])
    ax.set_yticklabels(['50', '100', '150', '200'])
    bars[0].set_color(green_light)
    bars[0].set_label(f'Unique ({size_unique}\%)')
    bars[1].set_label(f'Repeated ({size_repeated}\%)')
    ax.legend()
    ax.set_xlabel('Number of Repeated Transcription')
    ax.set_ylabel('Number of Samples (k)')
    plt.tight_layout()
    # plt.show()
    fig.savefig('images/data-repetitions.png')


if __name__ == '__main__':
    x = load_session()
    save_histogram(x)
