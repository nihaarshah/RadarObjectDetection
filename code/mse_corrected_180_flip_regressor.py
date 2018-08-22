# Calculate error between regressed angles and test correcting for the 180 degrees off 
# Use test_deep_angle_regressor to create the y_hat and y_test vectors
import numpy as np
m = len(y_hat)
y_error = np.empty([m])
for i in range(0,m):
	y_error[i] = np.absolute(y_hat[i]-y_test[i])
	if (165<y_error[i]<195) or (y_error[i]<15):
		y_error[i] = 0


# testing general error
ab = np.absolute(y_hat[:,0]-y_test)
sum = np.sum(ab)
sum/len(y_test)
#after mean absolute error after slack of 20 degrees about 180: 44 degrees
# mean abs error without the slack: 55 degrees

$\abs{\theta-\hat{\theta}}$