# import the necessary packages
import imutils
import cv2
import numpy as np
from collections import namedtuple
import numpy as np
import cv2

def pyramid(image, scale=1.5, minSize=(30, 30)):
	# yield the original image
	yield image

	# keep looping over the pyramid
	while True:
		# compute the new dimensions of the image and resize it
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)

		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break

		# yield the next image in the pyramid
		yield image

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in xrange(0, image.shape[0], stepSize):
		for x in xrange(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def orientations(image):

	rows,cols = image.shape
	for theta in range(0, 360, 45):
		M = cv2.getRotationMatrix2D((cols/2,rows/2),theta,1)
		image = cv2.warpAffine(image,M,(cols,rows))
		yield image

def find_nn(gt,pick):
	dist = []
	for (startX, startY, endX, endY) in pick:
		dist = np.append(dist,[np.sqrt(np.square(startX-gt[0,0])+np.square(startY-gt[0,1]))])
	idx = np.argmin(dist)
	return pick[idx]

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0,0])
	yA = max(boxA[1], boxB[0,1])
	xB = min(boxA[2], boxB[0,2])
	yB = min(boxA[3], boxB[0,3])

	# compute the area of intersection rectangle
	interArea = (xB - xA + 1) * (yB - yA + 1)

	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[0,2] - boxB[0,0] + 1) * (boxB[0,3] - boxB[0,1] + 1)

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)

	# return the intersection over union value
	return iou