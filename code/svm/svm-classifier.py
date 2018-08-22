import scipy.io as scio
import numpy as np


#Path to positive features matlab array
positive_path = '/Users/nihaar/Documents/4yp-code/data/taxi-rank2-car-patches/XTrainTrimmedPosThresholded.mat'

# Positive image samples 
pos_mat = scio.loadmat('/Users/nihaar/Documents/4yp/data/misc/taxi-rank2-car-patches/XTrainTrimmedPosThresholded.mat')
positive_patch = pos_mat['XTrainTrimmedPosThresholded']

# Negative images 
neg_mat = scio.loadmat('/Users/nihaar/Documents/4yp/data/misc/taxi-rank2-car-patches/XTrainTrimmedNegThresholded.mat')
negative_patch = neg_mat['XTrainTrimmedNegThresholded']

# Splitting training and test set in positive examples
x_train_pos=positive_patch[0:4360,:,:] 
x_test_pos=positive_patch[4360:5132,:,:] 

# Splitting training and test set in negative examples
x_train_neg=negative_patch[0:3460,:,:] # Again leaving out 200 images;
x_test_neg = negative_patch[3460:4071,:,:]

# Now y stuff

# training labels for these images
y_train_mat = scio.loadmat('/Users/nihaar/Documents/4yp/data/misc/taxi-rank2-car-patches/yTraining.mat')
y_train = y_train_mat['yTraining']
y_train = np.transpose(y_train)

# testing labels for the images
y_test_mat = scio.loadmat('/Users/nihaar/Documents/4yp/data/misc/taxi-rank2-car-patches/yTest.mat')
y_test = y_test_mat['yTest']
y_test = np.transpose(y_test)




# Concatenating the positive and negative training and testing sets
x_train=np.concatenate((x_train_pos,x_train_neg),axis=0)
x_test=np.concatenate((x_test_pos,x_test_neg),axis=0)

# Reshaped data tensor from 3d to 2d to suit SVM
x_train = np.reshape(x_train,[7820,1296],'F')
x_test = np.reshape(x_test,[1383,1296],'F')

# Reshaped target vector to suit SVM
y_test = y_test[:,0]
y_train = y_train[:,0]


# Randomly shuffling the rows of the feature matrix and corresponding target labels
y_train= np.reshape(y_train,[7820,1])   
xy_combined = np.concatenate((x_train, y_train), axis=1)
np.random.shuffle(xy_combined) 
m,n = xy_combined.shape
y_train = xy_combined[:,n-1]
x_train = xy_combined[:,0:n-1]

# Same for test data
y_test = np.reshape(y_test,[1383,1])
xy_combined = np.concatenate((x_test, y_test), axis=1)
np.random.shuffle(xy_combined) 
m,n = xy_combined.shape
y_test = xy_combined[:,n-1]
x_test = xy_combined[:,0:n-1]

# Starting te SVM stuff

import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.svm import LinearSVC



# Create a simple classifier
classifier = svm.LinearSVC()
classifier.fit(x_train, y_train)
y_score = classifier.decision_function(x_test)

# Print average pr value
from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_test, y_score)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))

# Plot PR curve
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

precision, recall, _ = precision_recall_curve(y_test, y_score)

plt.step(recall, precision, color='b', alpha=1,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))



# # Create a classifier: a support vector classifier
# classifier = svm.SVC(kernel='linear')

# # We learn the digits on the first half of the digits
# classifier.fit(x_train, y_train)

# # Now predict the value of the digit on the second half:
expected = y_test
predicted = y_score



# print("Classification report for classifier %s:\n%s\n"
#       % (classifier, metrics.classification_report(expected, predicted)))
# print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))




plt.show()