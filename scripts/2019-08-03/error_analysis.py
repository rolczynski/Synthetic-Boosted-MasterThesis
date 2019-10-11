import os
import pandas as pd
from matplotlib import pyplot as plt
import distances
import distances_detailed
import utils

direcotory = os.path.dirname(os.path.abspath(__file__))
logger = utils.create_logger(os.path.join(direcotory, 'error_analysis.log'))

dev_set = 'datasets/clarin/features/dev-clarin-normalized.hdf5'
train = utils.read(dev_set)
corpus = ' '.join(train.transcript)
known_words = set(corpus.split(' '))

base_fname = 'models/2019-06-24/00tune/evaluation.hdf5'
base_evaluation = utils.read(base_fname)
base_distances = distances.calculate(base_evaluation)
confusion_matrix, inserts, deletes = distances_detailed.calculate(base_evaluation)
distances_detailed.show(confusion_matrix)
distances_detailed.show_donut(inserts, deletes, confusion_matrix)

syn_fname = 'models/2019-08-03/05/evaluation.hdf5'
syn_evaluation = utils.read(syn_fname)
syn_distances = distances.calculate(syn_evaluation)

fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 4))
distances.plot(ax1, base_distances, title='Base')
distances.plot(ax2, syn_distances, title='Synthesized')
plt.show()

incorrect = lambda x: sum(x.wer != 0) / len(x) * 100
logger.info(f'Base Incorrect: {incorrect(base_evaluation):.0f}%')
logger.info(f'Synthesized Incorrect: {incorrect(syn_evaluation):.0f}%')
