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
import time as time
# classifier_svm = pickle.load(open('/Users/nihaar/Documents/4yp/data/weights/06-02-18/svm_hard_neg_mined_cross_val_model_aligned.pkl', 'rb'))
classifier_svm = pickle.load(open('/Users/nihaar/Documents/4yp/data/weights/06-02-18/svm_hard_neg_mined_cross_val_model_rotated.pkl', 'rb'))


# # get the data for svm rotated
x_test_svm, y_test_svm = load_svm_rotated_data.load_test_data_SVM_rotated()
start = time.time()
y_score = classifier_svm.predict(x_test_svm)
end = time.time()
print("Time taken %d",(end-start))
average_precision_svm = average_precision_score(y_test_svm, y_score)

# Test on the test set
predicted = y_score

# Calculate metrics
expected = y_test_svm

print('Average precision-recall score: {0:0.2f}'.format(average_precision_svm))

precision_svm, recall_svm, _ = precision_recall_curve(y_test_svm, y_score)

plt.step(recall_svm, precision_svm, color='b', alpha=1,where='post',label='SVM_rotated (AP:%f)'%(average_precision_svm))

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class P-R curve SVM classifier on rotated and aligned data:')

print(metrics.classification_report(expected, predicted))

print(metrics.confusion_matrix(expected, predicted))