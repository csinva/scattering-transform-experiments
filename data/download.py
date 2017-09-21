import h5py
import os
from os.path import join as oj

out = 'cifar_100'
if not os.path.exists(out):
    os.makedirs(out)
import tensorflow as tf
from tensorflow.contrib.keras.python.keras.datasets.cifar100 import load_data

print('downloading cifar 100....')
(x_train, y_train), (x_test, y_test) = load_data(
    label_mode='fine')  # Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`

print('shape', x_train.shape)
with h5py.File(oj(out, 'train.h5')) as f:
    f['X'] = x_train
    f['y'] = y_train
with h5py.File(oj(out, 'test.h5'), 'w') as f:
    f['X'] = x_test
    f['y'] = y_test