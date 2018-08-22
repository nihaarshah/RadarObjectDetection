# USAGE
# python sliding_window.py --image images/br-street-sweep_10.mat 

# import the necessary packages
from pyimagesearch.helpers import pyramid
from pyimagesearch.helpers import sliding_window
import argparse
import time
import cv2
import numpy as np
import cPickle
import numpy as np
from sklearn import datasets, svm, metrics
from sklearn.svm import LinearSVC
# from sklearn.metrics import average_precision_score
# from sklearn.metrics import precision_recall_curve


loaded_model = cPickle.load(open('/Users/nihaar/svm-classifier.pkl', 'rb'))
print('Loaded the SVM model...')

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the image and define the window width and height
image = cv2.imread(args["image"])
(winW, winH) = (41, 41)

# loop over the image pyramid
for resized in pyramid(image, scale=1.5):
	# loop over the sliding window for each layer of the pyramid
	for (x, y, window) in sliding_window(resized, stepSize=10, windowSize=(winW, winH)):
		# if the window does not meet our desired window size, ignore it
		if window.shape[0] != winH or window.shape[1] != winW:
			continue

		# THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
		# MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
		# WINDOW
		# x_test1 = 
		# loaded_model = cPickle.load(open('~/svm-classifier.pkl', 'rb'))
		# y_score = loaded_model.decision_function(x_test1)

		window = window[:,:,1]
		window = np.reshape(window,[-1,1681])
		# print(window.shape)
		y_score = loaded_model.decision_function(window)
		print(y_score)

		# since we do not have a classifier, we'll just draw the window
		clone = resized.copy()
		cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
		cv2.imshow("Window", clone)
		cv2.waitKey(1)
		time.sleep(0.25)