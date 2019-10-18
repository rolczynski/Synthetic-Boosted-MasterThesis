import dill
import numpy as np
from matplotlib import pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def load(file_path: str):
    with open(file_path, mode='rb') as file:
        return dill.load(file)


def plot_results(results_1: str, results_2: str):
    fig, ax = plt.subplots(figsize=(9, 4))

    data = load(results_1)
    epochs1, loss1, val_loss1 = np.array([epoch[:-1] for epoch in data]).T
    epochs1 = epochs1.astype(int) + 1
    min_index = np.argmin(val_loss1) + 1
    batch_results1 = np.array([epoch[-1] for epoch in data[1:min_index]]).flatten()
    epochs1, loss1, val_loss1 = epochs1[:min_index], loss1[:min_index], val_loss1[:min_index]

    data = load(results_2)
    epochs2, loss2, val_loss2 = np.array([epoch[:-1] for epoch in data]).T
    epochs2 += epochs1[-1] + 1
    batch_results2 = np.array([epoch[-1] for epoch in data]).flatten()
    batch_results = np.concatenate((batch_results1, batch_results2))

    epochs = np.concatenate((epochs1, epochs2)).astype(int)
    loss = np.concatenate((loss1, loss2))
    val_loss = np.concatenate((val_loss1, val_loss2))

    ax.plot(epochs, loss, label='CTC loss', color='black', alpha=0.8)
    ax.plot(epochs, val_loss, label=f'CTC dev loss', color='green', alpha=0.8)

    new_ax = ax.twiny()
    new_ax.plot(batch_results, alpha=0.3, linewidth=.05, color='black')
    new_ax.set_xticks([])

    for y in np.arange(5, 25, step=5):
        ax.axhline(y, xmin=0.0, xmax=1.0, alpha=0.2, color='black', linestyle='--', linewidth=1)

    ax.axvline(10, ymin=0.0, ymax=1.0, alpha=0.6, color='black', linestyle='--', linewidth=1)
    ax.text(7.5, 22, 'Phase I: Pretrain')
    ax.text(10.5, 22, 'Phase II: Tune')

    # ax.scatter(min_index+1, min_value, color='red', label=f'Min dev loss ({val_loss[min_index]:.2f})')

    ax.set_ylim(bottom=0, top=25)
    ax.set_xlabel('Epoch')
    ax.set_xticks(range(1, epochs[-1]+1))
    ax.set_ylabel('CTC Loss')
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('images/baseline.png')


if __name__ == '__main__':
    plot_results(results_1='data/results_1.bin',
                 results_2='data/results_2.bin')
