# This script flips a jpg image read from file upside down i.e. vertically.
import tensorflow as tf
import numpy as np
import cv2
# import sliding-window.images 

image_np_array = cv2.imread('imguint8_sweep39.jpg')

sess = tf.Session()
with sess.as_default():
	tensor_image_flipped = tf.image.flip_up_down(image_np_array)
	image_flipped_numpy = tensor_image_flipped.eval()
	cv2.imwrite('flipped_imguint8_sweep39.jpg',image_flipped_numpy)