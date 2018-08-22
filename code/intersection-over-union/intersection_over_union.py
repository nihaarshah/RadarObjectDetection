# USAGE
# python intersection_over_union.py
# This may be useful for evaulating the accuracy of our bounding box predictions 
# in the case of detection.

# import the necessary packages
from collections import namedtuple
import numpy as np
import cv2

# define the `Detection` object
Detection = namedtuple("Detection", ["image_path", "gt1","gt2", "pred"])

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	# compute the area of intersection rectangle
	interArea = (xB - xA + 1) * (yB - yA + 1)

	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)

	# return the intersection over union value
	return iou

# define the list of example detections
# 39, 63, 203, 112
examples = [
	Detection("image_0002.jpg", [0, 10, 200, 110],[95,40,200,180], [54, 66, 198, 114])]
	# Detection("image_0016.jpg", [49, 75, 203, 125], [42, 78, 186, 126]),
	# Detection("image_0075.jpg", [31, 69, 201, 125], [18, 63, 235, 135]),
	# Detection("image_0090.jpg", [50, 72, 197, 121], [54, 72, 198, 120]),
	# Detection("image_0120.jpg", [35, 51, 196, 110], [36, 60, 180, 108])]

# loop over the example detections
for detection in examples:
	# load the image
	image = cv2.imread(detection.image_path)

	# draw the ground-truth bounding box along with the predicted
	# bounding box
	cv2.rectangle(image, tuple(detection.gt1[:2]), 
		tuple(detection.gt1[2:]), (0, 255, 0), 2)
	cv2.rectangle(image, tuple(detection.gt2[:2]), 
		tuple(detection.gt2[2:]), (0, 255, 0), 2)
	cv2.rectangle(image, tuple(detection.pred[:2]), 
		tuple(detection.pred[2:]), (0, 0, 255), 2)

	# compute the intersection over union and display it
	iou1 = bb_intersection_over_union(detection.gt1, detection.pred)
	iou2 = bb_intersection_over_union(detection.gt2, detection.pred)
	cv2.putText(image, "IoU: {:.4f}".format(iou1), (10, 30),
		cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
	print("{}: {:.4f}".format(detection.image_path, iou1))
	cv2.putText(image, "IoU: {:.4f}".format(iou2), (10, 30),
		cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
	print("{}: {:.4f}".format(detection.image_path, iou2))
	# show the output image
	cv2.imshow("Image", image)
	cv2.waitKey(0)