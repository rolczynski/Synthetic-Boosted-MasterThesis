import os
os.chdir('/mnt/lun1/rolczynski')

import sys
sys.path.append('/mnt/lun1/rolczynski/DeepSpeech-Keras')

from scripts.evaluate import calculate_units
from source.deepspeech import DeepSpeech
from source.configuration import ModelConfiguration
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# i = ...
# model = DeepSpeech.get_model(**dict(
#     name='deepspeech',
#     units=i,
#     is_gpu=False
# ))

# model = DeepSpeech.get_model(**dict(
#     name='deepspeech-custom',
#     layers=[
#         {'name': 'expand_dims', 'axis': -1},
#         {'name': 'ZeroPadding2D', 'padding': (7, 0)},
#         {'name': 'Conv2D', 'filters': i, 'kernel_size': (15, 80)},
#         {'name': 'squeeze', 'axis': 2},
#         {'name': 'ReLU', 'max_value': 20},
#         {'name': 'Dropout', 'rate': 0.1},
#         {'name': 'LSTM', 'units': i, 'return_sequences': True},
#         {'name': 'LSTM', 'units': i, 'return_sequences': True},
#         {'name': 'LSTM', 'units': i, 'return_sequences': True}
#     ],
#     input_dim=80,
#     output_dim=26,
#     is_gpu=False
# ))


config = ModelConfiguration('models/2019-08-13/03/configuration.yaml')
model = DeepSpeech.get_model(is_gpu=False, **config.model)

units = calculate_units(model) / 1e6
print(units)

pass
