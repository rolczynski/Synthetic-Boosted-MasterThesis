import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from utils import read_activations
from sklearn.model_selection import train_test_split
np.random.seed(123)


def pipeline(index):
    rich_samples_activations = read_activations('models/2019-07-06/00tune/evaluation.hdf5', output_index=index)
    synthesized_samples_activations = read_activations('models/2019-07-06/00tune/evaluation-syn.hdf5', output_index=index)

    rich_full = np.concatenate(rich_samples_activations, axis=0)
    synthesized_full = np.concatenate(synthesized_samples_activations, axis=0)

    subset_size = 10000
    rich = rich_full[np.random.choice(len(rich_full), subset_size)]
    synthesized = synthesized_full[np.random.choice(len(synthesized_full), subset_size)]

    rich_mean = rich.mean(axis=0) / len(rich)
    synthesized_mean = synthesized.mean(axis=0) / len(synthesized)
    # distance = euclidean(rich_mean, sythesized_mean)
    distance = np.sum(np.absolute(rich_mean - synthesized_mean))

    X = np.concatenate([rich, synthesized], axis=0)
    y = np.concatenate([np.zeros(len(rich)), np.ones(len(synthesized))], axis=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = Sequential()
    model.add(Dense(1, input_dim=650, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(X_train, y_train, batch_size=64, epochs=50, validation_data=[X_test, y_test])
    return distance, history


fname = 'MasterThesis/models/2019-07-06/results.bin'
distances = []
acc = []
val_acc = []
for i in [5, 6, 7, 8, 9]:
    distance, history = pipeline(i)
    distances.append(distance)
    acc.append(history.history['acc'])
    val_acc.append(history.history['val_acc'])
pickle.dump([distances, acc, val_acc], open(fname, 'wb'))
