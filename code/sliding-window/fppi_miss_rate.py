# USAGE
# python sliding_window.py --image images/br-street-sweep_10.mat 
# This script loads a jpg of the radar sweep (which was created using the dot_mat_to_jpg.m script)
# and calculates an svm score(representative of the distance from the hyperplane)
# for a sliding window across the entire radar sweep. The radar sweep is then 
# rotated through several angles and the whole score calculation done once again.

# import the necessary packages
from pyimagesearch.helpers import pyramid
from pyimagesearch.helpers import sliding_window
from pyimagesearch.helpers import find_nn
from pyimagesearch.helpers import bb_intersection_over_union
from pyimagesearch.nms import non_max_suppression_fast
import argparse
import time
import cv2
import numpy as np
import scipy.io as sio
# import pickle
import numpy as np
from sklearn import datasets, svm, metrics
from sklearn.svm import LinearSVC
from keras.models import load_model
# import matplotlib.pyplot as plt

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


deep_model = load_model('/Users/nihaar/Documents/4yp/data/weights/05-02-18/dnn_rotated_model.h5')

img_path = '/Users/nihaar/Documents/4yp/data/hand-labelled-lamb-flag/sweeps_tensor.mat'
gt_path = '/Users/nihaar/Documents/4yp/data/hand-labelled-lamb-flag/'
neg_mat1 = scio.loadmat(img_path)
sweep_tensor = neg_mat1['sweeps_tensor']

mat_contents = sio.loadmat('/Users/nihaar/Documents/4yp/data/hand-labelled-lamb-flag/gt_cell.mat')
gt_cell = mat_contents['gt_cell']

(winW, winH) = (36, 36)
# detections = np.empty((0,2))

sum_fppi = np.zeros((1,10))
sum_miss = np.zeros((1,10))
counter = 0
# loop over the sliding window for each layer of the pyramid
for frame in range(0,90):
	counter += 1
	# GT = np.matrix([[270, 366, 306, 402],[220,  439,  256, 475],[202,476,238,512]])
	GT = np.matrix(gt_cell[0,frame])
	# Initializing matrix containing detection cooridnates
	detections = np.empty((0,4))
	# Loading matrix of all radar sweeps and normalizing them
	image = sweep_tensor[frame,:,:]
	image = image/127
	fppi = []
	miss_rate = []
	tp = []	
	gt_mis = []
	for thresh in [x * 0.1 for x in range(0, 10, 1)]:
		detections = np.empty((0,4))
		for (x, y, window) in sliding_window(image, stepSize=10, windowSize=(winW, winH)):
			# if the window does not meet our desired window size, ignore it
			if window.shape[0] != winH or window.shape[1] != winW:
				continue
			
			window = np.reshape(window,[1,36,36,1])

			y_test_probability = deep_model.predict(window)

			# we'll just draw the window
			clone = image.copy()
			start_x = x 
			end_x = x + winW
			start_y = y 
			end_y = y + winH

			# make a matrix of all the detections for this threshold
			if (y_test_probability > thresh) and (np.sqrt(np.square((x+18)-100)+np.square((y+18)-100))>15):
				detections = np.append(detections,np.array([(start_x,start_y,end_x,end_y)]),axis=0)
		
		detections = detections.astype(int)
		clone = image.copy()
		
		# NMS filtered detections pick contains rows of format startx,starty,endx,endy
		pick = non_max_suppression_fast(detections, 0.3)
		l,_ = pick.shape
		# print("The threshold is",thresh,"and the number of fp are",l)
		n,_ = GT.shape
		thresh_overlap = .2
		gt_missed = 0.0
		true_pos = 0
		fp = 0
		# For each ground truth in an image search for nearest prediction and if they overlap count as true positive else add to miss
		for gt in GT:
			# find iou of gt with its nearest neighbour in the format [startx, starty ...]
			bbnn = find_nn(gt,pick)
			# print("bbnn",bbnn)
			iou = bb_intersection_over_union(bbnn,gt)
			if iou < thresh_overlap:
				gt_missed += 1.0
			else:
				true_pos += 1
		fp = l - true_pos
		# calculating over all thresholds		
		fppi = np.append(fppi,fp)
		missr = gt_missed/float(n)
		miss_rate = np.append(miss_rate, missr)
		tp = np.append(tp,true_pos)
		gt_mis = np.append(gt_mis,gt_missed)

	sum_fppi = sum_fppi + fppi
	sum_miss = sum_miss + miss_rate
	print("fppi for this pic is",sum_fppi)
	print("miss rate is for the pic is",sum_miss)
np.savetxt('sum_miss.txt',sum_miss/counter,fmt='%d')
np.savetxt('sum_fppi.txt',sum_fppi/counter,fmt='%d')

	# print("true positives are",tp)
	# print("gt missed are these",gt_mis)
	# plt.plot(miss_rate,fppi)
	# plt.show()