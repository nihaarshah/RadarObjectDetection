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



def load_train_data_DNN():
#Joining all real feature matrices from various datasets into one big matrix
	positive_patch = np.concatenate((pp1,pp2,pp3,pp5),axis = 0)
	negative_patch = np.concatenate((np1,np3),axis =0)
	# Preparing training data
	m,n = positive_patch.shape
	i,j = negative_patch.shape

	# Trim 41 x 41 dimension to 36 x 36 to suit architecture
	pp_tens = np.reshape(positive_patch,[m,41,41])
	np_tens = np.reshape(negative_patch,[i,41,41])
	# Trimming down the patches
	pp_tens = pp_tens[:,2:38,2:38]
	np_tens = np_tens[:,2:38,2:38]

	pp_np_tens = np.concatenate((pp_tens,np_tens),axis =0)
	y_train_pos = np.ones(m)
	y_train_neg = np.zeros(i)
	y = np.concatenate((y_train_pos,y_train_neg))

	return pp_np_tens,y	

def load_test_data_DNN():

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

	return x_test,y_test