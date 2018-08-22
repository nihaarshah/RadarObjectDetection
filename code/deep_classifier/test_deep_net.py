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
from sklearn.model_selection import train_test_split
from helpers.load_test_data import load_test_data_DNN, load_train_data_DNN
from keras.models import load_model
from sklearn import datasets, svm, metrics
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

pp_np_tens, y = load_train_data_DNN()
X_test, y_test = load_test_data_DNN()

#Splitting training data randomly for the purpose of validation (NB there is no testing happening in this script)
X_train, X_val, y_train, y_val = train_test_split(pp_np_tens, y, test_size=0.33, random_state=42)

X_train = X_train.astype('float32') 
X_val = X_val.astype('float32')
X_train = np.reshape(X_train, (len(X_train), 36,36,1))
X_val = np.reshape(X_val, (len(X_val), 36,36,1))
X_test = X_test.astype('float32')
X_test = np.reshape(X_test, (len(X_test), 36,36,1))

# Load model
classifier = load_model('/Users/nihaar/Documents/4yp/data/weights/08-01-18/softmax_model.h5')

#Predict scores
y_hat = classifier.predict(X_test)

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