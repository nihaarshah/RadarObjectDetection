# This script loads training and test data for rotated patches for DNN and SVM
# This script is to load test data from lamb-and-flag 1 new-detections as of 12th Jan
# in the format of a m x w x h matrix where m = number of features/patches and w&h the dimensions
# Tra

from keras import backend as K
import numpy as np
import scipy.io as scio
from random import shuffle



# Prepare the data
#Path to positive features matlab array
positive_path1 = '/Users/nihaar/Documents/4yp/data/detections-with-bounding-box-coordinates/taxi-rank-2/rotated_real_features.mat'
positive_path2 = '/Users/nihaar/Documents/4yp/data/detections-with-bounding-box-coordinates/outside-uni-parks/rotated_real_features.mat'
positive_path3 = '/Users/nihaar/Documents/4yp/data/detections-with-bounding-box-coordinates/nuffield-college-1/rotated_real_features.mat'
positive_path4 = '/Users/nihaar/Documents/4yp/data/detections-with-bounding-box-coordinates/lamb-and-flag1/rotated_real_features.mat'
positive_path5 = '/Users/nihaar/Documents/4yp/data/detections-with-bounding-box-coordinates/broad-street/rotated_real_features.mat'

negative_path1 = '/Users/nihaar/Documents/4yp/data/new-detections/taxi-rank-2/fake_features_matrix.mat'
negative_path2 = '/Users/nihaar/Documents/4yp/data/new-detections/outside-uni-parks-1/fake_features_matrix.mat'
negative_path3 = '/Users/nihaar/Documents/4yp/data/new-detections/nuffleld college 1/fake_features_matrix.mat'
negative_path4 = '/Users/nihaar/Documents/4yp/data/new-detections/lamb and flag 1/fake_features_matrix.mat'
negative_path5 = '/Users/nihaar/Documents/4yp/data/new-detections/broad street/fake_features_matrix.mat'

aec_neg1_path = '/Users/nihaar/Documents/4yp/data/detections-with-bounding-box-coordinates/broad-street/fake_features_matrix.mat'
aec_neg2_path = '/Users/nihaar/Documents/4yp/data/new-detections/lamb and flag 1/fake_features_matrix.mat'
aec_neg3_path = '/Users/nihaar/Documents/4yp/data/detections-with-bounding-box-coordinates/outside-uni-parks/fake_features_matrix.mat'
aec_neg4_path = '/Users/nihaar/Documents/4yp/data/detections-with-bounding-box-coordinates/nuffield-college-1/fake_features_matrix.mat'
aec_neg5_path = '/Users/nihaar/Documents/4yp/data/detections-with-bounding-box-coordinates/taxi-rank-2/fake_features_matrix.mat'


# Positive image samples 
pos_mat1 = scio.loadmat(positive_path1)
pp1 = pos_mat1['rotated_real_features']

pos_mat2 = scio.loadmat(positive_path2)
pp2 = pos_mat2['rotated_real_features']

pos_mat3 = scio.loadmat(positive_path3)
pp3 = pos_mat3['rotated_real_features']

pos_mat4 = scio.loadmat(positive_path4)
pp4 = pos_mat4['rotated_real_features']

pos_mat5 = scio.loadmat(positive_path5)
pp5 = pos_mat5['rotated_real_features']

# Negative images 
neg_mat1 = scio.loadmat(aec_neg1_path)
np1 = neg_mat1['fake_features_matrix']

neg_mat2 = scio.loadmat(aec_neg2_path)
np2 = neg_mat2['fake_features_matrix']

neg_mat3 = scio.loadmat(aec_neg3_path)
np3 = neg_mat3['fake_features_matrix']

neg_mat4 = scio.loadmat(aec_neg4_path)
np4 = neg_mat4['fake_features_matrix']

neg_mat5 = scio.loadmat(aec_neg5_path)
np5 = neg_mat5['fake_features_matrix']

# neg_mat1 = scio.loadmat(negative_path1)
# np1 = neg_mat1['fake_features_matrix']

# neg_mat2 = scio.loadmat(negative_path2)
# np2 = neg_mat2['fake_features_matrix']

# neg_mat3 = scio.loadmat(negative_path3)
# np3 = neg_mat3['fake_features_matrix']

# neg_mat4 = scio.loadmat(negative_path4)
# np4 = neg_mat4['fake_features_matrix']

# neg_mat5 = scio.loadmat(negative_path5)
# np5 = neg_mat5['fake_features_matrix']


# Prepare data for regression
# Prepare data
# rot_path_1 = '/Users/nihaar/Documents/4yp/data/detections-with-bounding-box-coordinates/taxi-rank-2/rotated_real_features-tr2.mat'
# angles_path_1 = '/Users/nihaar/Documents/4yp/data/detections-with-bounding-box-coordinates/taxi-rank-2/angles-tr2.mat'

# rot_mat1 = scio.loadmat(rot_path_1)
# rot1 = rot_mat1['rotated_real_features']
# rot1_train = rot1[0:24000,:]
# rot1_test = rot1[24001:25305,:]

# angles1 = scio.loadmat(angles_path_1)
# ang1 = angles1['angles']
# ang1_train = ang1[0,0:24000]
# ang1_test = ang1[0,24001:25305]


# Prepare data for autoencoder pretraining. The negatives in this are 57x57 from the detections-bounding-box folder. As that
# folder has larger number of negatives. The positives are the same as rotated data as that is 3x more than organic positives.
aec_neg1_path = '/Users/nihaar/Documents/4yp/data/detections-with-bounding-box-coordinates/broad-street/fake_features.mat'
aec_neg2_path = '/Users/nihaar/Documents/4yp/data/new-detections/lamb and flag 1/fake_features_matrix.mat'
aec_neg3_path = '/Users/nihaar/Documents/4yp/data/detections-with-bounding-box-coordinates/outside-uni-parks/fake_features.mat'
aec_neg4_path = '/Users/nihaar/Documents/4yp/data/detections-with-bounding-box-coordinates/nuffield-college-1/fake_features.mat'
aec_neg5_path = '/Users/nihaar/Documents/4yp/data/detections-with-bounding-box-coordinates/taxi-rank-2/fake_features.mat'


# Negative images 
neg_mat1_aec = scio.loadmat(aec_neg1_path)
np1_aec = neg_mat1_aec['fake_features_matrix']

neg_mat3_aec = scio.loadmat(aec_neg3_path)
np3_aec = neg_mat3_aec['fake_features_matrix']

neg_mat4_aec = scio.loadmat(aec_neg4_path)
np4_aec = neg_mat4_aec['fake_features_matrix']

neg_mat5_aec = scio.loadmat(aec_neg5_path)
np5_aec = neg_mat5_aec['fake_features_matrix']

def load_train_data_SVM_rotated():

	
	# Testing dataset next
	positive_patch_train = np.concatenate((pp1,pp2,pp3,pp5),axis=0)
	negative_patch_train = np.concatenate((np1,np3),axis=0)
	q,_ = np.shape(positive_patch_train)
	s,_ = np.shape(negative_patch_train)

	negative_patch_train = np.reshape(negative_patch_train,[s,41,41])
	negative_patch_train = negative_patch_train[:,2:38,2:38]
	negative_patch_train = np.reshape(negative_patch_train,[s,1296])

	x_train=np.concatenate((positive_patch_train,negative_patch_train),axis=0)

	# testing labels for the images
	y_train_pos = np.ones(q)
	y_train_neg = np.zeros(s)
	y_train = np.concatenate((y_train_pos,y_train_neg))
	y_train = np.transpose(y_train)
	# Ignore the following split if you wish to do K fold (K>1) cross val
	# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.33, random_state=42)
	return x_train,y_train

def load_train_data_SVM_rotated_HNM():
	
	negative_path1 = '/Users/nihaar/Documents/4yp/data/new-detections/taxi-rank-2/fake_features_matrix.mat'
	neg_mat1 = scio.loadmat(negative_path1)
	negpat1 = neg_mat1['fake_features_matrix']
	negative_path4 = '/Users/nihaar/Documents/4yp/data/new-detections/lamb and flag 1/fake_features_matrix.mat'
	neg_mat4 = scio.loadmat(negative_path4)
	np4 = neg_mat4['fake_features_matrix']

	#  dataset next
	positive_patch_train = np.concatenate((pp1,pp2,pp3),axis=0)
	
	p,_ = np.shape(positive_patch_train)
	s,_ = np.shape(negpat1)
	
	negpat1 = np.reshape(negpat1,[s,41,41])
	negpat1 = negpat1[:,2:38,2:38]
	negpat1 = np.reshape(negpat1,[s,1296])

	negative_patch_train = negpat1[0:100,:]
	m,_ = np.shape(negative_patch_train)
	q,_ = np.shape(np4)
	
	np4 = np.reshape(np4,[q,41,41])
	np4 = np4[:,2:38,2:38]
	np4 = np.reshape(np4,[q,1296])

	x_val = np.concatenate((np4,pp5),axis=0)

	t,_ = np.shape(pp5)
	y_val = np.concatenate((np.zeros(q),np.ones(t)))

	# x_val = np.reshape(x_val,[q,41,41])
	# x_val = x_val[:,2:38,2:38]
	# x_val = np.reshape(x_val,[s,1296])

	x_train=np.concatenate((positive_patch_train,negative_patch_train),axis = 0)

	# testing labels for the images
	y_train_pos = np.ones(p)
	y_train_neg = np.zeros(m)
	y_train = np.concatenate((y_train_pos, y_train_neg))
	y_train = np.transpose(y_train)
	# Ignore the following split if you wish to do K fold (K>1) cross val
	# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.33, random_state=42)
	return x_train,y_train, x_val, y_val

def load_test_data_SVM_rotated():

	# Testing dataset next
	positive_patch_test = pp4
	negative_patch_test = np4
	q,_ = np.shape(positive_patch_test)
	s,_ = np.shape(negative_patch_test)
	negative_patch_test = np.reshape(negative_patch_test,[s,41,41])
	negative_patch_test = negative_patch_test[:,2:38,2:38]
	negative_patch_test = np.reshape(negative_patch_test,[s,1296])

	x_test=np.concatenate((positive_patch_test,negative_patch_test),axis=0)

	# testing labels for the images
	y_test_pos = np.ones(q)
	y_test_neg = np.zeros(s)
	y_test = np.concatenate((y_test_pos,y_test_neg))
	y_test = np.transpose(y_test)

	return x_test,y_test

def load_train_data_DNN_rotated():
#Joining all real feature matrices from various datasets into one big matrix
	positive_patch = np.concatenate((pp1,pp2,pp3,pp5),axis = 0)
	negative_patch = np.concatenate((np1,np3),axis =0)
	# Preparing training data
	m,n = positive_patch.shape
	i,j = negative_patch.shape

	pp_tens = np.reshape(positive_patch,[m,36,36])
	np_tens = np.reshape(negative_patch,[i,41,41])
	np_tens = np_tens[:,2:38,2:38]
	pp_np_tens = np.concatenate((pp_tens,np_tens),axis =0)
	y_train_pos = np.ones(m)
	y_train_neg = np.zeros(i)
	y = np.concatenate((y_train_pos,y_train_neg))

	return pp_np_tens,y	

def load_train_data_AEC_rotated():
#Joining all real feature matrices from various datasets into one big matrix
	positive_patch = np.concatenate((pp1,pp2,pp3,pp4,pp5),axis = 0)
	negative_patch = np.concatenate((np1_aec,np3_aec,np4_aec,np5_aec),axis =0)
	# Preparing training data
	m,n = positive_patch.shape
	i,j = negative_patch.shape

	pp_tens = np.reshape(positive_patch,[m,36,36])
	np_tens = np.reshape(negative_patch,[i,57,57])
	np_tens = np_tens[:,2:38,2:38]
	pp_np_tens = np.concatenate((pp_tens,np_tens),axis =0)
	y_train_pos = np.ones(m)
	y_train_neg = np.zeros(i)
	y = np.concatenate((y_train_pos,y_train_neg))

	return pp_np_tens,y	

def load_test_data_DNN_rotated():

	# Preparing the test data
	pp_test = pp4
	np_test = np4
	x,_ = pp_test.shape
	y,_ = np_test.shape
	pp_tens_test = np.reshape(pp_test,[x,36,36])
	np_tens_test = np.reshape(np_test,[y,41,41])
	# Trimming down the patches
	np_tens_test = np_tens_test[:,2:38,2:38]
	x_test = np.concatenate((pp_tens_test,np_tens_test),axis =0)
	x_test = x_test.astype('float32')
	x_test = np.reshape(x_test, (len(x_test), 36,36,1))
	y_test_pos = np.ones(x)
	y_test_neg = np.zeros(y)
	y_test = np.concatenate((y_test_pos,y_test_neg))

	return x_test,y_test

def load_train_data_DNN_rotated_regression_train():
#Joining all real feature matrices from various datasets into one big matrix
	positive_patch = rot1_train
	# Preparing training data
	m,n = positive_patch.shape

	pp_tens = np.reshape(positive_patch,[m,36,36])
	# Transposing such that the image index remains in its old position thus 0 and rows and columns get swapped thus 
	# 0 where 1 and 1 where 0
	pp_tens = np.transpose(pp_tens,(0,2,1))
	y = ang1_train

	return pp_tens,y

def load_train_data_DNN_rotated_regression_test():
#Joining all real feature matrices from various datasets into one big matrix
	positive_patch = rot1_test
	# Preparing training data
	m,n = positive_patch.shape

	pp_tens = np.reshape(positive_patch,[m,36,36])
	# Transposing such that the image index remains in its old position thus 0 and rows and columns get swapped thus 
	# 0 where 1 and 1 where 0
	pp_tens = np.transpose(pp_tens,(0,2,1))
	y = ang1_test

	return pp_tens,y	