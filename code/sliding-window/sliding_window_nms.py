# USAGE
# python sliding_window.py --image images/br-street-sweep_10.mat 
# This script loads a jpg of the radar sweep (which was created using the dot_mat_to_jpg.m script)
# and calculates an svm score(representative of the distance from the hyperplane)
# for a sliding window across the entire radar sweep. The radar sweep is then 
# rotated through several angles and the whole score calculation done once again.

# import the necessary packages
# from pyimagesearch.helpers import pyramid
from pyimagesearch.helpers import sliding_window
# from pyimagesearch.helpers import orientations
from pyimagesearch.nms import non_max_suppression_fast
import argparse
import time
import cv2
import numpy as np
# import pickle
import numpy as np
# from sklearn import datasets, svm, metrics
# from sklearn.svm import LinearSVC
from keras.models import load_model
# import matplotlib.pyplot as plt

from keras.models import Model
from keras import backend as K

import scipy.io as scio
import tensorflow as tf
# from keras.callbacks import TensorBoard
# from random import shuffle
# import keras.initializers as ki
# import sklearn.metrics as skm
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import average_precision_score
# from sklearn import datasets, svm, metrics

# cv2.waitKey(8000)

deep_model = load_model('/Users/nihaar/Documents/4yp/data/weights/05-02-18/dnn_rotated_model.h5')


img_path = '/Users/nihaar/Documents/4yp/data/hand-labelled-lamb-flag/sweeps_tensor.mat'
neg_mat1 = scio.loadmat(img_path)

sweep_tensor = neg_mat1['sweeps_tensor']

frame_size = 561

(winW, winH) = (36, 36)
# detections = np.empty((0,2))

#threads=4
#`concurrent(threads, while()) 
# loop over the sliding window for each layer of the pyramid
for frame in range(88,89):
	start = time.time()
	detections = np.empty((0,4))
	image = sweep_tensor[frame,:,:]
	image = image/127
	image = cv2.flip(image,0)
	image = cv2.flip(image,1)
	for (x, y, window) in sliding_window(image, stepSize=10, windowSize=(winW, winH)):
		# if the window does not meet our desired window size, ignore it
		if window.shape[0] != winH or window.shape[1] != winW:
			continue
		
		window = np.reshape(window,[1,36,36,1])

		y_test_probability = deep_model.predict(window)

		# since we do not have a classifier, we'll just draw the window
		# clone = image.copy()
		start_x = x 
		end_x = x + winW
		start_y = y 
		end_y = y + winH

		if (y_test_probability > 0.93) and (np.sqrt(np.square((x+18)-100)+np.square((y+18)-100))>20):
			detections = np.append(detections,np.array([(start_x,start_y,end_x,end_y)]),axis=0)
			# cv2.rectangle(clone, (x, y), (x + winW, y + winH), (255, 0, 0), 1)
		# else:
		# 	cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 1)
		# cv2.imshow("Window", clone)
		# cv2.waitKey(1)
		# time.sleep(.0001)

	detections = detections.astype(int)
	clone = image.copy()
	# l,_ = detections.shape
	pick = non_max_suppression_fast(detections, .3)
	l,_ = pick.shape
	print("number of filtered detections",l)
	for (startX, startY, endX, endY) in pick:
		# print(startY,startX,endX,endY)
		cv2.rectangle(clone, (startX, startY), (endX, endY), (255, 0, 0), 1)
	for i in range(0,l):

		# x = x.astype(int)
		# y = y.astype(int)
		cv2.rectangle(clone, (x, y), (x + winW, y + winH), (255, 0, 0), 1)
	cv2.imshow("Frame:%s"%(1), clone)
	end = time.time()
	print("Frame:%s"%(frame),"Time taken: %s"%(end-start))
	cv2.waitKey(0)