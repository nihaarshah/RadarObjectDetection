# This script follows from the "train_softmax_more_data" script that had trained a deep network after 
# unsupervised pretraining. This simply reloads the saved weights after training and then tests using 
# confusion matrices etc. The idea is to make the training and testing modular.

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten
from keras.models import Model
from keras import backend as K
import numpy as np
import scipy.io as scio
import tensorflow as tf
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from random import shuffle
import keras.initializers as ki
import sklearn.metrics as skm
from keras.models import load_model
from sklearn import datasets, svm, metrics
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
import sys
import time
sys.path.insert(0, '/Users/nihaar/Documents/4yp/4yp/code/helpers/')
import load_CNN_data_rotated 


X_test, y_test = load_CNN_data_rotated.load_test_data_DNN_rotated()


X_test = X_test.astype('float32')
X_test = np.reshape(X_test, (len(X_test), 36,36,1))

# Load model
classifier = load_model('/Users/nihaar/Documents/4yp/data/weights/05-02-18/dnn_rotated_model.h5')

#Predict scores
start = time.time()
y_hat = classifier.predict(X_test)
end = time.time()
print("Time taken by CNN is %d",(end-start))
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