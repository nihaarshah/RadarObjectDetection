import pickle
import scipy.io as scio
import numpy as np
import sys
sys.path.insert(0, '/Users/nihaar/Documents/4yp/4yp/code/helpers/')
from load_test_data import load_test_data_SVM, load_test_data_DNN
import load_data_rotated 
from sklearn import datasets, svm, metrics
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from keras.models import load_model

# Load test data
x_test_dnn, y_test_dnn = load_test_data_DNN()




classifier_dnn_pretrained = load_model('/Users/nihaar/Documents/4yp/data/weights/25-02-18/softmax_model_with_pretraining_without_freezing_layers.h5')
# predict 
y_hat_dnn_with_pt = classifier_dnn_pretrained.predict(x_test_dnn)

# PR curve
average_precision_dnn_with_pt = average_precision_score(y_test_dnn, y_hat_dnn_with_pt)
precision_dnn_with_pt, recall_dnn_with_pt, _ = precision_recall_curve(y_test_dnn, y_hat_dnn_with_pt)

plt.step(recall_dnn_with_pt, precision_dnn_with_pt, color='r', alpha=1.0,where='post',label='Pretrained (Greedily) CNN (AP:%f)'%(average_precision_dnn_with_pt))

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class P-R curve comparison of with and without pretraining data:')


# DNN test
# classifier_dnn_not_pt = load_model('/Users/nihaar/Documents/4yp/data/weights/25-02-18/softmax_model_without_pretraining.h5')
classifier_dnn_not_pt = load_model('/Users/nihaar/Documents/4yp/data/weights/25-02-18/softmax_model_with_pretraining_without_freezing_layers.h5')
y_hat_dnn_not_pt = classifier_dnn_not_pt.predict(x_test_dnn)

# PR curve
average_precision_dnn_not_pt = average_precision_score(y_test_dnn, y_hat_dnn_not_pt)
precision_dnn_not_pt, recall_dnn_not_pt, _ = precision_recall_curve(y_test_dnn, y_hat_dnn_not_pt)

plt.step(recall_dnn_not_pt, precision_dnn_not_pt, color='b', alpha=1.0,where='post',label='pretrained (Not-greedily) CNN (AP:%f)'%(average_precision_dnn_not_pt))

plt.legend()
plt.show()
