# USAGE
# python sliding_window.py --image images/br-street-sweep_10.mat 
# This script loads a jpg of the radar sweep (which was created using the dot_mat_to_jpg.m script)
# and calculates an svm score(representative of the distance from the hyperplane)
# for a sliding window across the entire radar sweep. The radar sweep is then 
# rotated through several angles and the whole score calculation done once again.

# import the necessary packages
from pyimagesearch.helpers import pyramid
from pyimagesearch.helpers import sliding_window
from pyimagesearch.helpers import orientations
import argparse
import time
import cv2
import numpy as np
# import pickle
import numpy as np
from sklearn import datasets, svm, metrics
from sklearn.svm import LinearSVC
from keras.models import load_model


from keras.models import Model
from keras import backend as K

import scipy.io as scio
import tensorflow as tf
from keras.callbacks import TensorBoard
from random import shuffle
import keras.initializers as ki
import sklearn.metrics as skm
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn import datasets, svm, metrics

# from sklearn.metrics import average_precision_score
# from sklearn.metrics import precision_recall_curve

#Loading the SVM trained model 
# loaded_model = pickle.load(open('/Users/nihaar/balanced_training_set_model.pkl', 'rb'))
# print('Loaded the SVM model...')
#Loading the Deep network model
deep_model = load_model('/Users/nihaar/Documents/4yp/data/weights/15-01-18/softmax_model.h5')

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="Path to the image")
# args = vars(ap.parse_args())
img_path = '/Users/nihaar/Documents/4yp/4yp/code/sliding-window/br-street-sweeps.mat'
# load the image and define the window width and height
# image = cv2.imread(args["image"])
neg_mat1 = scio.loadmat(img_path)
sweep_tensor = neg_mat1['img']
# image = image[59,:,:]
# image = image/127

(winW, winH) = (36, 36)
# angle = (0,45,90,135)
# image = image[:,:,1]
# image = image/128
# ori = orientations(image)
detections = np.empty((0,2))


# loop over the sliding window for each layer of the pyramid
for frame in range(10,43):
	detections = np.empty((0,2))
	image = sweep_tensor[frame,:,:]
	image = image/127
	for (x, y, window) in sliding_window(image, stepSize=10, windowSize=(winW, winH)):
		# if the window does not meet our desired window size, ignore it
		if window.shape[0] != winH or window.shape[1] != winW:
			continue

		# THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
		# MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
		# WINDOW
		# x_test1 = 
		# loaded_model = cPickle.load(open('~/svm-classifier.pkl', 'rb'))
		# y_score = loaded_model.decision_function(x_test1)
		
		window = np.reshape(window,[1,36,36,1])
		# window = np.reshape(window,[-1,1681])
		# print(window.shape)

		# y_test_probability = loaded_model.predict_proba(window)  # For SVM
		y_test_probability = deep_model.predict(window)
		# print(y_test_probability)

		# since we do not have a classifier, we'll just draw the window
		clone = image.copy()
		if (y_test_probability > 0.8):
			detections = np.append(detections,np.matrix([[x,y]]),axis=0)
			cv2.rectangle(clone, (x, y), (x + winW, y + winH), (255, 0, 0), 1)
		else:
			cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 1)
		cv2.imshow("Window", clone)
		cv2.waitKey(1)
		time.sleep(.0001)
	# detections = np.matrix([[80, 70], [150, 70], [140, 80]])
	detections = detections.astype(int)
	clone = image.copy()
	l,_ = detections.shape
	for i in range(0,l):
		x = detections[i,0]
		y = detections[i,1]
		x = x.astype(int)
		y = y.astype(int)
		cv2.rectangle(clone, (x, y), (x + winW, y + winH), (255, 0, 0), 1)
	cv2.imshow("Frame:%s"%(frame+1), clone)
	cv2.waitKey(0)
# print(detections)