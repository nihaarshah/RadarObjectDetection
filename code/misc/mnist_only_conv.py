rom __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.
  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.
  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

# The encoder
  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([3, 3, 1, 16])
    b_conv1 = bias_variable([16])
    h_conv1 = tf.nn.relu(conv2d_stride2(x_image, W_conv1) + b_conv1)


  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([3, 3, 16, 8])
    b_conv2 = bias_variable([8])
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)


# At this stage the dimensions is  11 x 11 x 8


#Decoder

# 2nd Deconv layer (reverse of the third conv layer)
  with tf.name_scope('deconv3'):
    W_deconv3 = weight_variable([3,3,8,16])  
    b_deconv3 = bias_variable([16])
    h_deconv3 = tf.nn.relu(deconv2dlayer1(h_conv2, W_deconv3)+ b_deconv3)

# 1st deconv layer (reverse of the 2nd conv layer)
  with tf.name_scope('deconv3'):
    W_deconv3 = weight_variable([3,3,16,1])  
    b_deconv3 = bias_variable([1])
    h_deconv3 = tf.nn.relu(deconv2d_stride2(h_pool3, W_deconv3)+ b_deconv3)


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def conv2d_stride2(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='VALID')

def deconv2dlayer1(x, W):
  """ deconv layer inverse of a conv2d"""
  return tf.layers.conv2d_transpose(x,W,[13,13],strides=(1, 1),padding='VALID',data_format='channels_last',activation=None)

def deconv2d_stride2(x,W):
  """ deconv layer inverse of a conv2d"""
  return tf.layers.conv2d_transpose(x,W,[1,13,13,16],strides=(2, 2),padding='VALID',data_format='channels_last',activation=None)

def deconv2d_no_padding(x):
    """conv2d just this time there is no padding so the convolved output in the 3 x 3 filter case will have dimesnsions: (H-2 x W-2 x C)"""
    return tf.layers.conv2d_transpose(x,8,[3,3],strides=(1, 1),padding='same',data_format='channels_last',activation=None)

def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
  



def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # Build the graph for the deep net
  y_conv, keep_prob = deepnn(x)

  with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy)

  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  graph_location = tempfile.mkdtemp()
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
      batch = mnist.train.next_batch(50)
      if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        print('step %d, training accuracy %g' % (i, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)