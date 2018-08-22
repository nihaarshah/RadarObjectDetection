# USAGE
# python sliding_window.py --image images/br-street-sweep_10.mat 
# This script loads a jpg of the radar sweep (which was created using the dot_mat_to_jpg.m script)
# and calculates an svm score(representative of the distance from the hyperplane)
# for a sliding window across the entire radar sweep. The radar sweep is then 
# rotated through several angles and the whole score calculation done once again.

from pyimagesearch.helpers import sliding_window
# from pyimagesearch.helpers import orientations
from pyimagesearch.nms import non_max_suppression_fast
import argparse
import time
import cv2
from multiprocessing import Pool
import threading
import numpy as np
from keras.models import load_model
from keras.models import Model
from keras import backend as K
import scipy.io as scio
import tensorflow as tf

lock = threading.Lock()
model_path = '/Users/nihaar/Documents/4yp/data/weights/05-02-18/dnn_rotated_model.h5'
deep_model = load_model(model_path)
img_path = '/Users/nihaar/Documents/4yp/data/hand-labelled-lamb-flag/sweeps_tensor.mat'
neg_mat1 = scio.loadmat(img_path)
sweep_tensor = neg_mat1['sweeps_tensor']
frame_size = 561
(winW, winH) = (36, 36)

class CNN:
	def __init__(self, model_path):

	    self.cnn_model = load_model(model_path)
	    self.cnn_model.predict(np.zeros([1,36,36,1])) # warmup
	    self.session = K.get_session()
	    self.graph = tf.get_default_graph()
	    self.graph.finalize() # finalize
	    print("finished init")

cnn = CNN(model_path)

#threads=4
#`concurrent(threads, while()) 
# loop over the sliding window for each layer of the pyramid
def f(frame):
	print("entered f %d",frame )
	detections = np.empty((0,4))
	image = sweep_tensor[frame,:,:]
	image = image/127
	image = cv2.flip(image,0)
	image = cv2.flip(image,1)
	for (x, y, window) in sliding_window(image, stepSize=10, windowSize=(winW, winH)):
		print("entered loop 1")
		# if the window does not meet our desired window size, ignore it
		if window.shape[0] != winH or window.shape[1] != winW:
			continue
		window = np.reshape(window,[1,36,36,1])
		# print("window is chosen")
		# lock.acquire()
		# print("Lock acquired")
		y_test_probability = deep_model.predict(window)
		# y_test_probability = 0.5
		# print("work done with lock")
		# lock.release()
		# print("prediction made")
		start_x = x 
		end_x = x + winW
		start_y = y 
		end_y = y + winH
		if (y_test_probability > 0.90) and (np.sqrt(np.square((x+18)-100)+np.square((y+18)-100))>20):
			detections = np.append(detections,np.array([(start_x,start_y,end_x,end_y)]),axis=0)
			# cv2.rectangle(clone, (x, y), (x + winW, y + winH), (255, 0, 0), 1)
		# else:
		# 	cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 1)
		# cv2.imshow("Window", clone)
		# cv2.waitKey(1)
		# time.sleep(.0001)
	print("exited loop 1")
	detections = detections.astype(int)
	clone = image.copy()
	# l,_ = detections.shape
	pick = non_max_suppression_fast(detections, 0.3)
	# l,_ = pick.shape
	# print("number of filtered detections",l)
	# for (startX, startY, endX, endY) in pick:
	# 	# print(startY,startX,endX,endY)
	# 	cv2.rectangle(clone, (startX, startY), (endX, endY), (255, 0, 0), 1)
	# for i in range(0,l):

	# 	x = x.astype(int)
	# 	y = y.astype(int)
	# 	cv2.rectangle(clone, (x, y), (x + winW, y + winH), (255, 0, 0), 1)
	# cv2.imshow("Frame:%s"%(1), clone)
	print("Frame:%s"%(frame),"Time taken: %s"%(end-start))
	cv2.waitKey(1)

start = time.time()
a = np.arange(0,3)
print(a)
if __name__ == '__main__':
    p = Pool(4)
    o = p.map(f,a)
        # print(o)
end = time.time()
print(end-start)