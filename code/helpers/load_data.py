# This script is to load test data from lamb-and-flag 1 new-detections as of 12th Jan
# in the format of a m x n matrix where m = number of features and n = pixel intensity of 
# each feauture.

from keras import backend as K
import numpy as np
import scipy.io as scio
from random import shuffle

positive_path4 = '/Users/nihaar/Documents/4yp/data/new-detections/lamb and flag 1/real_features_matrix.mat'

negative_path4 = '/Users/nihaar/Documents/4yp/data/new-detections/lamb and flag 1/fake_features_matrix.mat'

pos_mat4 = scio.loadmat(positive_path4)
pp4 = pos_mat4['real_features_matrix']

neg_mat4 = scio.loadmat(negative_path4)
np4 = neg_mat4['fake_features_matrix']


# Preparing the test data
pp_test = pp4
np_test = np4
x,_ = pp_test.shape
y,_ = np_test.shape
pp_tens_test = np.reshape(pp_test,[x,41,41])
np_tens_test = np.reshape(np_test,[y,41,41])
# Trimming down the patches
pp_tens_test = pp_tens_test[:,2:38,2:38]
np_tens_test = np_tens_test[:,2:38,2:38]
x_test = np.concatenate((pp_tens_test,np_tens_test),axis =0)
x_test = x_test.astype('float32')
x_test = np.reshape(x_test, (len(x_test), 36,36,1))
y_test_pos = np.ones(x)
y_test_neg = np.zeros(y)
y_test = np.concatenate((y_test_pos,y_test_neg))