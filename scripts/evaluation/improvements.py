import os
import pickle
import distances
import utils

direcotory = os.path.dirname(os.path.abspath(__file__))
logger = utils.create_logger(os.path.join(direcotory, 'error_analysis.log'))

base_fname = 'models/2019-06-24/00tune/final-evaluation.hdf5'
base_evaluation = utils.read(base_fname)
base_distances = distances.calculate(base_evaluation)

syn_fname = 'models/2019-08-13/10/final-evaluation.hdf5'
syn_evaluation = utils.read(syn_fname)
syn_distances = distances.calculate(syn_evaluation)

fname = 'MasterThesis/scripts/evaluation/improvements.bin'
pickle.dump([base_distances, syn_distances], open(fname, 'wb'))
