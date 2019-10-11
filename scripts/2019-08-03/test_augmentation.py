import os
import pandas as pd
from matplotlib import pyplot as plt
import distances
import utils

direcotory = os.path.dirname(os.path.abspath(__file__))
logger = utils.create_logger(os.path.join(direcotory, 'test_augmentation.log'))

dev_fname = 'datasets/clarin/features/dev-clarin-normalized.hdf5'
dev_evaluation = utils.read(dev_fname)

syn_fname = 'models/2019-08-03/base/evaluation-syn.hdf5'
syn_evaluation = utils.read(syn_fname)
correct = syn_evaluation.wer == 0
baseline = syn_evaluation[correct].copy()
indices = range(1, 10)
synthesizeds = [utils.read(f'models/2019-08-03/base/evaluation-syn-{i}.hdf5')[correct].copy() for i in indices]
syn_distances = [distances.calculate(syn) for syn in synthesizeds]

# synthesized, *_ = synthesizeds
# syn_distance, *_ = syn_distances
# first_group_indices = (syn_distance.word == 1) & (syn_distance.char == 1)
# first_group_indices = first_group_indices[first_group_indices].index
# first_group = synthesized.loc[first_group_indices]

fig, axes = plt.subplots(3, 3, figsize=(4*3, 4*3))
for ax, syn_distance in zip(axes.flatten(), syn_distances):
    distances.plot(ax, syn_distance)
plt.show()

# syn_distances = distances.calculate(syn_evaluation)
# distances.show(syn_distances)
