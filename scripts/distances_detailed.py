import numpy as np
import pandas as pd
from collections import Counter


def update_(confusion_matrix: np.ndarray, to_substitute: dict, alphabet):
    """ Update the confusion matrix. """
    for correct_char, wrong_chars in to_substitute.items():
        correct_char_label = alphabet.label_from_string(correct_char)
        wrong_chars_labels = [alphabet.label_from_string(char) for char in wrong_chars]

        for wrong_char_label in wrong_chars_labels:
            confusion_matrix[correct_char_label, wrong_char_label] += 1


def calculate(evaluation: pd.DataFrame):
    import utils
    from source.text import Alphabet
    from source.metric import edit_distance, naive_backtrace, decode_

    ALPHABET = Alphabet('DeepSpeech-Keras/tests/models/base/alphabet.txt')

    inserts, deletes = Counter(), Counter()
    confusion_matrix = np.zeros([ALPHABET.size, ALPHABET.size], dtype=int)
    results = evaluation[['transcript', 'prediction']].copy()

    for index, original, prediction in results.itertuples():
        distance, edit_distance_matrix, backtrace = edit_distance(source=prediction, destination=original)
        best_path = naive_backtrace(backtrace)
        to_delete, to_insert, to_substitute = decode_(best_path, prediction, original)
        update_(confusion_matrix, to_substitute, ALPHABET)
        inserts.update(to_insert)
        deletes.update(to_delete)
    return confusion_matrix, inserts, deletes
