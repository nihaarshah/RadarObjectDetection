import scipy.io as scio
import cv2

img_path = '/Users/nihaar/Documents/4yp/data/hand-labelled-lamb-flag/sweeps_tensor.mat'
neg_mat1 = scio.loadmat(img_path)
sweep_tensor = neg_mat1['sweeps_tensor']

# for frame in range(23,26):
# 	image = sweep_tensor[frame,:,:]
# 	image = image/127
# 	clone = image.copy()
# 	cv2.imshow("Framea:%s"%(1), clone)
# 	cv2.waitKey(2000)



for frame in range(23,26):
	image = sweep_tensor[frame,:,:]
	image = image/127
	clone = image.copy()
	cv2.imshow("Frame:", clone)
	cv2.waitKey(2000)