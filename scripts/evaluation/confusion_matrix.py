import pickle
import utils
import distances_detailed

fname = 'MasterThesis/scripts/evaluation/confusion_matrix.bin'
references = utils.read('models/2019-08-13/10/final-evaluation.hdf5')
confusion_matrix, inserts, deletes = distances_detailed.calculate(references)
pickle.dump([confusion_matrix, inserts, deletes], open(fname, 'wb'))
