# Prepare data for regression
import sys
sys.path.insert(0, '/Users/nihaar/Documents/4yp/4yp/code/helpers/')
import numpy as np
import scipy.io as scio	

rot_path_1 = '/Users/nihaar/Documents/4yp/data/detections-with-bounding-box-coordinates/taxi-rank-2/rotated_real_features-tr2.mat'
angles_path_1 = '/Users/nihaar/Documents/4yp/data/detections-with-bounding-box-coordinates/taxi-rank-2/angles-tr2.mat'

rot_mat1 = scio.loadmat(rot_path_1)
rot1 = rot_mat1['rotated_real_features']
rot1_train = rot1[0:24000,:]
rot1_test = rot1[24001:25305,:]

angles1 = scio.loadmat(angles_path_1)
ang1 = angles1['angles']
ang1_train = ang1[0,0:24000]
ang1_test = ang1[0,24001:25305]



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