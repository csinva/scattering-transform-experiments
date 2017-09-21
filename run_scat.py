import tensorflow as tf
import os, sys, time, subprocess, h5py, argparse, logging
import numpy as np
from os.path import join as oj
from libs.scattering import scattering

with h5py.File('data/cifar_100/train.h5') as f:
    ims = f['X'][0:5]

print('ims.shape before', ims.shape)
ims = np.transpose(ims, (0, 3, 1, 2))  # convert NHWC -> NCHW
print('ims.shape after', ims.shape)
im_shape = ims.shape[1:]


# can only run on a gpu
# requires NCHW format (cuDNN default - tf is NHWC)
placeholder = tf.placeholder(tf.float32, (None,) + im_shape)
# M, N: input image size
M, N = placeholder.shape.as_list()[-2:]
print("M", M, "N", N)
# J: number of layers
scat = scattering.Scattering(M=M, N=N, J=1)(placeholder)


def extract_features(placeholder, model, ims):
    print('ims.shape', ims.shape)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    t = time.time()
    output = sess.run(model, feed_dict={placeholder: ims})
    print('features.shape', output.shape)
    return output

with tf.device("/cpu:0"):
    features = extract_features(placeholder=placeholder, model=scat, ims=ims)