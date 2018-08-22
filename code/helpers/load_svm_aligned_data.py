from keras import backend as K
import numpy as np
import scipy.io as scio
from random import shuffle



# Prepare the data
#Path to positive features matlab array
positive_path1 = '/Users/nihaar/Documents/4yp/data/new-detections/taxi-rank-2/real_features_matrix.mat'
positive_path2 = '/Users/nihaar/Documents/4yp/data/new-detections/outside-uni-parks-1/real_features_matrix.mat'
positive_path3 = '/Users/nihaar/Documents/4yp/data/new-detections/nuffleld college 1/real_features_matrix.mat'
positive_path4 = '/Users/nihaar/Documents/4yp/data/new-detections/lamb and flag 1/real_features_matrix.mat'
positive_path5 = '/Users/nihaar/Documents/4yp/data/new-detections/broad street/real_features_matrix.mat'

negative_path1 = '/Users/nihaar/Documents/4yp/data/new-detections/taxi-rank-2/fake_features_matrix.mat'
negative_path2 = '/Users/nihaar/Documents/4yp/data/new-detections/outside-uni-parks-1/fake_features_matrix.mat'
negative_path3 = '/Users/nihaar/Documents/4yp/data/new-detections/nuffleld college 1/fake_features_matrix.mat'
negative_path4 = '/Users/nihaar/Documents/4yp/data/new-detections/lamb and flag 1/fake_features_matrix.mat'
negative_path5 = '/Users/nihaar/Documents/4yp/data/new-detections/broad street/fake_features_matrix.mat'

# Positive image samples 
pos_mat1 = scio.loadmat(positive_path1)
pp1 = pos_mat1['real_features_matrix']

pos_mat2 = scio.loadmat(positive_path2)
pp2 = pos_mat2['real_features_matrix']

pos_mat3 = scio.loadmat(positive_path3)
pp3 = pos_mat3['real_features_matrix']

pos_mat4 = scio.loadmat(positive_path4)
pp4 = pos_mat4['real_features_matrix']

pos_mat5 = scio.loadmat(positive_path5)
pp5 = pos_mat5['real_features_matrix']

# Negative images 
neg_mat1 = scio.loadmat(negative_path1)
np1 = neg_mat1['fake_features_matrix']

neg_mat2 = scio.loadmat(negative_path2)
np2 = neg_mat2['fake_features_matrix']

neg_mat3 = scio.loadmat(negative_path3)
np3 = neg_mat3['fake_features_matrix']

neg_mat4 = scio.loadmat(negative_path4)
np4 = neg_mat4['fake_features_matrix']

neg_mat5 = scio.loadmat(negative_path5)
np5 = neg_mat5['fake_features_matrix']

def load_train_data_SVM():

	# Testing dataset next
	positive_patch_train = np.concatenate((pp1,pp2,pp3,pp5),axis=0)
	negative_patch_train = np.concatenate((np1,np3),axis=0)

	x_train=np.concatenate((positive_patch_train,negative_patch_train),axis=0)

	q,_ = np.shape(positive_patch_train)
	s,_ = np.shape(negative_patch_train)

	# testing labels for the images
	y_train_pos = np.ones(q)
	y_train_neg = np.zeros(s)
	y_train = np.concatenate((y_train_pos,y_train_neg))
	y_train = np.transpose(y_train)
	# Ignore the following split if you wish to do K fold (K>1) cross val
	# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.33, random_state=42)
	return x_train,y_train

def load_test_data_SVM():

	# Testing dataset next
	positive_patch_test = pp4
	negative_patch_test = np4

	x_test=np.concatenate((positive_patch_test,negative_patch_test),axis=0)
	q,_ = np.shape(positive_patch_test)
	s,_ = np.shape(negative_patch_test)

	# testing labels for the images
	y_test_pos = np.ones(q)
	y_test_neg = np.zeros(s)
	y_test = np.concatenate((y_test_pos,y_test_neg))
	y_test = np.transpose(y_test)

	return x_test,y_test