import pickle
import scipy.io as scio
import numpy as np
import sys
sys.path.insert(0, '/Users/nihaar/Documents/4yp/4yp/code/helpers/')
import load_CNN_data_rotated 
import load_CNN_data_aligned 
from sklearn import datasets, svm, metrics
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from keras.models import load_model



# DNN test
x_test_dnn, y_test_dnn = load_CNN_data_rotated.load_test_data_DNN_rotated()

classifier_dnn = load_model('/Users/nihaar/Documents/4yp/data/weights/05-02-18/dnn_rotated_model.h5')
# classifier_dnn = load_model('/Users/nihaar/Documents/4yp/data/weights/08-01-18/softmax_model.h5')
y_hat_dnn = classifier_dnn.predict(x_test_dnn)

# PR curve
average_precision_dnn = average_precision_score(y_test_dnn, y_hat_dnn)
precision_dnn, recall_dnn, _ = precision_recall_curve(y_test_dnn, y_hat_dnn)

plt.step(recall_dnn, precision_dnn, color='b', alpha=1.0,where='post',label='CNN_rotated (AP:%f)'%(average_precision_dnn))



# CNN aligned
x_test_dnn, y_test_dnn = load_CNN_data_aligned.load_test_data_DNN()
classifier_dnn = load_model('/Users/nihaar/Documents/4yp/data/weights/08-01-18/softmax_model.h5')
y_hat_dnn = classifier_dnn.predict(x_test_dnn)

# PR curve
average_precision_dnn = average_precision_score(y_test_dnn, y_hat_dnn)
precision_dnn, recall_dnn, _ = precision_recall_curve(y_test_dnn, y_hat_dnn)

plt.step(recall_dnn, precision_dnn, color='r', alpha=1.0,where='post',label='CNN_aligned (AP:%f)'%(average_precision_dnn))


plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class P-R curve CNN classifier on rotated and aligned data:')
plt.legend()
plt.show()
