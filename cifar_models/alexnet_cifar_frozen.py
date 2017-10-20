################################################################################
#Michael Guerzhoy and Davi Frossard, 2016
#AlexNet implementation in TensorFlow, with weights
#Details: 
#http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#
#With code from https://github.com/ethereon/caffe-tensorflow
#Model from  https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
#Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow
#
#
################################################################################

from numpy import *
import os
import numpy as np
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random
import argparse
import sys
import tempfile
import matplotlib.pyplot as plt


import tensorflow as tf

from sklearn import linear_model, datasets
import h5py
filename = "train.h5"
f = h5py.File(filename, 'r')
train_x = np.array(f['X'])
temp_y = np.array([y[0] for y in f['y']]).flatten()
train_y = np.zeros((len(temp_y), 100))
train_y[np.arange(len(temp_y)), temp_y] = 1
f.close()
testfile = "test.h5"
f2 = h5py.File(testfile, 'r')
test_x = np.array(f2['X'])
temp_y = np.array([y[0] for y in f2['y']]).flatten()
test_y = np.zeros((len(temp_y), 100))
test_y[np.arange(len(temp_y)), temp_y] = 1
f2.close()

xdim = train_x.shape[1:]



################################################################################
#Read Image, and change to BGR


# im1 = (imread("laska.png")[:,:,:3]).astype(float32)
# im1 = im1 - mean(im1)
# im1[:, :, 0], im1[:, :, 2] = im1[:, :, 2], im1[:, :, 0]

# im2 = (imread("poodle.png")[:,:,:3]).astype(float32)
# im2[:, :, 0], im2[:, :, 2] = im2[:, :, 2], im2[:, :, 0]


################################################################################

# (self.feed('data')
#         .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
#         .lrn(2, 2e-05, 0.75, name='norm1')
#         .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
#         .conv(5, 5, 256, 1, 1, group=2, name='conv2')
#         .lrn(2, 2e-05, 0.75, name='norm2')
#         .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
#         .conv(3, 3, 384, 1, 1, name='conv3')
#         .conv(3, 3, 384, 1, 1, group=2, name='conv4')
#         .conv(3, 3, 256, 1, 1, group=2, name='conv5')
#         .fc(4096, name='fc6')
#         .fc(4096, name='fc7')
#         .fc(1000, relu=False, name='fc8')
#         .softmax(name='prob'))

#In Python 3.5, change this to:
net_data = load(open("bvlc_alexnet.npy", "rb"), encoding="latin1").item()
#net_data = load("bvlc_alexnet.npy").item()

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    
    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups =  tf.split(input, group, 3)   #tf.split(3, group, input)
        kernel_groups = tf.split(kernel, group, 3)  #tf.split(3, group, kernel) 
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)          #tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])

#ADDED CODE START
def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
#ADDED CODE END

x = tf.placeholder(tf.float32, (None,) + xdim)


#conv1
#conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
conv1W = tf.Variable(net_data["conv1"][0])
conv1b = tf.Variable(net_data["conv1"][1])
conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
conv1 = tf.nn.relu(conv1_in)

#lrn1
#lrn(2, 2e-05, 0.75, name='norm1')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn1 = tf.nn.local_response_normalization(conv1,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

#maxpool1
#max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


#conv2
#conv(5, 5, 256, 1, 1, group=2, name='conv2')
k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
conv2W = tf.Variable(net_data["conv2"][0])
conv2b = tf.Variable(net_data["conv2"][1])
conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv2 = tf.nn.relu(conv2_in)


#lrn2
#lrn(2, 2e-05, 0.75, name='norm2')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn2 = tf.nn.local_response_normalization(conv2,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

#maxpool2
#max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#conv3
#conv(3, 3, 384, 1, 1, name='conv3')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
conv3W = tf.Variable(net_data["conv3"][0])
conv3b = tf.Variable(net_data["conv3"][1])
conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv3 = tf.nn.relu(conv3_in)

#conv4
#conv(3, 3, 384, 1, 1, group=2, name='conv4')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
conv4W = tf.Variable(net_data["conv4"][0])
conv4b = tf.Variable(net_data["conv4"][1])
conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv4 = tf.nn.relu(conv4_in)


#conv5
#conv(3, 3, 256, 1, 1, group=2, name='conv5')
k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
conv5W = tf.Variable(net_data["conv5"][0])
conv5b = tf.Variable(net_data["conv5"][1])
conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv5 = tf.nn.relu(conv5_in)

# #maxpool5
# #max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
# #k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
# k_h = 1; k_w = 1; s_h = 1; s_w = 1; padding = 'VALID'

# maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
keep_prob = tf.placeholder(tf.float32)

#fc6
#fc(4096, name='fc6')
#fc6W = tf.Variable(net_data["fc6"][0])
#fc6b = tf.Variable(net_data["fc6"][1])
fc6W = weight_variable([256, 1024])
fc6b = bias_variable([1024])
fc6 = tf.nn.relu_layer(tf.reshape(conv5, [-1, int(prod(conv5.get_shape()[1:]))]), fc6W, fc6b)
fc6_drop = tf.nn.dropout(fc6, keep_prob)

#fc7
#fc(4096, name='fc7')
#fc7W = tf.Variable(net_data["fc7"][0])
#fc7b = tf.Variable(net_data["fc7"][1])
fc7W = weight_variable([1024, 1024])
fc7b = bias_variable([1024])

fc7 = tf.nn.relu_layer(fc6_drop, fc7W, fc7b)
fc7_drop = tf.nn.dropout(fc7, keep_prob)

#fc8
#fc(1000, relu=False, name='fc8')
#fc8W = tf.Variable(net_data["fc8"][0])
#fc8b = tf.Variable(net_data["fc8"][1])
fc8W = weight_variable([1024, 100])
fc8b = bias_variable([100])

fc8 = tf.nn.xw_plus_b(fc7_drop, fc8W, fc8b)


#prob
#softmax(name='prob'))
prob = tf.nn.softmax(fc8)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

#output = sess.run(prob, feed_dict = {x:train_x}).reshape(len(train_x), 256)


y_ = tf.placeholder(tf.float32, [None, 100])

with tf.name_scope('loss'):
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                          logits=fc8)
cross_entropy = tf.reduce_mean(cross_entropy)


with tf.name_scope('adam_optimizer'):
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope('accuracy'):
  correct_prediction = tf.equal(tf.argmax(fc8, 1), tf.argmax(y_, 1))
  correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

graph_location = tempfile.mkdtemp()
print('Saving graph to: %s' % graph_location)
train_writer = tf.summary.FileWriter(graph_location)
train_writer.add_graph(tf.get_default_graph())
saver = tf.train.Saver()

l_acc = []
t_acc = []
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(10000):
    indices = np.random.randint(0, 50000,size = 1000)
    batch = [train_x[indices], train_y[indices]]
    if i % 1 == 0:
      train_accuracy = accuracy.eval(feed_dict={
          x: batch[0], y_: batch[1], keep_prob:1.0})
      print('step %d, training accuracy %g' % (i, train_accuracy))
      l_acc.append(train_accuracy)
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob:0.9})
    if i % 250 == 0:
      test_accuracy = accuracy.eval(feed_dict={
        x: test_x, y_: test_y, keep_prob: 1.0})
      print('test accuracy %g' % test_accuracy)
      t_acc.append(test_accuracy)

  save_path = saver.save(sess, "cifar.ckpt")
plt.plot(l_acc)
plt.savefig('l_acc.png')
plt.close()
plt.plot(t_acc)
plt.savefig('t_acc.png')
plt.close()