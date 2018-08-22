# This script combines power of SVM and DNN so that we pass the detections of the SVM 
# to the DNN 

import pickle
import scipy.io as scio
import numpy as np
import sys
sys.path.insert(0, '/Users/nihaar/Documents/4yp/4yp/code/helpers/')
from load_test_data import load_test_data_SVM
import load_data_rotated 
from keras.models import load_model
from sklearn import datasets, svm, metrics
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

loaded_model = pickle.load(open('/Users/nihaar/Documents/4yp/data/weights/06-02-18/svm_hard_neg_mined_cross_val_model_aligned.pkl', 'rb'))

x_test,y_test = load_data_rotated()


y_test_probabilities = loaded_model.predict_proba(x_test)
short_listed_idx = []
for i in range (0,len(y_test)):
	if y_test_probabilities[i,1] > 0.3:
		short_listed_idx.append(i)

x_short_listed_test = np.empty([0,1681])
y_short_listed_test = np.empty([0,])
for z in short_listed_idx:
	row = x_test[z,:]
	row = np.reshape(row,[1,1681])
	x_short_listed_test = np.append(x_short_listed_test,row,axis=0)
	row2 = y_test[z]
	y_short_listed_test = np.append(y_short_listed_test,row2)

# average_precision = average_precision_score(y_test, y_test_predictions_high_recall_p3)
pp_test = x_short_listed_test
x,_ = pp_test.shape
pp_tens_test = np.reshape(pp_test,[x,41,41])
# Trimming down the patches
pp_tens_test = pp_tens_test[:,2:38,2:38]
x_test = pp_tens_test
x_test = x_test.astype('float32')
x_test = np.reshape(x_test, (len(x_test), 36,36,1))
y_test = y_short_listed_test

classifier = load_model('/Users/nihaar/Documents/4yp/data/weights/08-01-18/softmax_model.h5')

#Predict scores
y_hat = classifier.predict(x_test)

# PR curve
average_precision = average_precision_score(y_test, y_hat)
precision, recall, _ = precision_recall_curve(y_test, y_hat)

plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))
plt.show()

# Confusion matrix 
y_hat = y_hat[:] > 0.5
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(y_test, y_hat)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, y_hat))