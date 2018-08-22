import pickle
import scipy.io as scio
import numpy as np
import sys
sys.path.insert(0, '/Users/nihaar/Documents/4yp/4yp/code/helpers/')
import load_svm_rotated_data 
import load_svm_aligned_data 
import pickle
from sklearn import datasets, svm, metrics
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from keras.models import load_model

# classifier_svm = pickle.load(open('/Users/nihaar/Documents/4yp/data/weights/06-02-18/svm_hard_neg_mined_cross_val_model_aligned.pkl', 'rb'))
classifier_svm = pickle.load(open('/Users/nihaar/Documents/4yp/data/weights/06-02-18/svm_hard_neg_mined_cross_val_model_rotated.pkl', 'rb'))


# # get the data for svm rotated
x_test_svm, y_test_svm = load_svm_rotated_data.load_test_data_SVM_rotated()

y_score = classifier_svm.decision_function(x_test_svm)
average_precision_svm = average_precision_score(y_test_svm, y_score)

print('Average precision-recall score: {0:0.2f}'.format(average_precision_svm))

precision_svm, recall_svm, _ = precision_recall_curve(y_test_svm, y_score)

plt.step(recall_svm, precision_svm, color='b', alpha=1,where='post',label='SVM_rotated (AP:%f)'%(average_precision_svm))

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class P-R curve SVM classifier on rotated and aligned data:')


# SVM aligned test
# Get the data for svm aligned
x_test_svm, y_test_svm = load_svm_aligned_data.load_test_data_SVM()
classifier_svm = pickle.load(open('/Users/nihaar/Documents/4yp/data/weights/06-02-18/svm_hard_neg_mined_cross_val_model_aligned.pkl', 'rb'))

y_score = classifier_svm.decision_function(x_test_svm)
average_precision_svm = average_precision_score(y_test_svm, y_score)

print('Average precision-recall score: {0:0.2f}'.format(average_precision_svm))

precision_svm, recall_svm, _ = precision_recall_curve(y_test_svm, y_score)

plt.step(recall_svm, precision_svm, color='r', alpha=1,where='post',label='SVM_aligned (AP:%f)'%(average_precision_svm))

plt.legend()
plt.show()



# DNN test
x_test_dnn, y_test_dnn = load_data_rotated.load_test_data_DNN_rotated()
# x_test_dnn, y_test_dnn = load_test_data.load_test_data_DNN()
classifier_dnn = load_model('/Users/nihaar/Documents/4yp/data/weights/05-02-18/dnn_rotated_model.h5')
# classifier_dnn = load_model('/Users/nihaar/Documents/4yp/data/weights/08-01-18/softmax_model.h5')
y_hat_dnn = classifier_dnn.predict(x_test_dnn)

# PR curve
average_precision_dnn = average_precision_score(y_test_dnn, y_hat_dnn)
precision_dnn, recall_dnn, _ = precision_recall_curve(y_test_dnn, y_hat_dnn)

plt.step(recall_dnn, precision_dnn, color='r', alpha=1.0,where='post',label='CNN (AP:%f)'%(average_precision_dnn))

plt.legend()
plt.show()
