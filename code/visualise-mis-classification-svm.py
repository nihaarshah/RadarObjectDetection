# Extract mis classifications in the test set to Visualise 
# Training which involves hard negative mining on the validation set
import scipy.io as scio
import numpy as np
from sklearn import datasets, svm, metrics
from sklearn.svm import LinearSVC
import pickle
from sklearn.model_selection import train_test_split
import sys
sys.path.insert(0, '/Users/nihaar/Documents/4yp/4yp/code/helpers/')
import load_test_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

x_train,y_train = load_test_data.load_train_data_SVM()

i,_ = x_train.shape
loaded_model = pickle.load(open('/Users/nihaar/Documents/4yp/4yp/Experiment-Results/05-02-18/HNM/svm_hard_neg_mined_cross_val_model_aligned.pkl','rb'))
y_score = loaded_model.decision_function(x_train)
y_train_probabilities = loaded_model.predict_proba(x_train)


# Vector of indices of false-positives in the validation set
fp = []
for j in range(0,i):
	if y_train[j] == 0 and y_score[j] >= 0:
  		# if y_train_probabilities[j,1] > 0.7:
  		fp.append(j)
fn = []
for j in range(0,i):
	if y_train[j] == 1 and y_score[j] <= 0:
  		# if y_train_probabilities[j,1] > 0.7:
  		fn.append(j)

tp_easy = []
for j in range(0,i):
	if y_train[j] == 1 and y_score[j] >= 0:
  		if y_train_probabilities[j,1] > 0.7:
  			tp_easy.append(j)

tp_hard = []
for j in range(0,i):
	if y_train[j] == 1 and y_score[j] >= 0:
  		if 0.5 < y_train_probabilities[j,1] < 0.7:
  			tp_hard.append(j)

tn_easy = []
for j in range(0,i):
	if y_train[j] == 0 and y_score[j] <= 0:
  		if y_train_probabilities[j,0] > 0.7:
  			tn_easy.append(j)

tn_hard = []
for j in range(0,i):
	if y_train[j] == 0 and y_score[j] <= 0:
  		if 0.5 < y_train_probabilities[j,0] < 0.7:
  			tn_hard.append(j)

# Create a matrix of false negative features and correasponding labels
fn_mat = np.empty([0,1681])
fn_mat = x_train[fn,:]
# y_fn = np.zeros(len(fp))

# False positive matrix
fp_mat = np.empty([0,1681])
fp_mat = x_train[fp,:]

# easy true pos features 
tp_easy_mat = np.empty([0,1681])
tp_easy_mat = x_train[tp_easy,:]

# hard true pos features 
tp_hard_mat = np.empty([0,1681])
tp_hard_mat = x_train[tp_hard,:]

# easy true neg features 
tn_easy_mat = np.empty([0,1681])
tn_easy_mat = x_train[tn_easy,:]

# hard true neg features 
tn_hard_mat = np.empty([0,1681])
tn_hard_mat = x_train[tn_hard,:]

cd '/Users/nihaar/Documents/4yp/data/classification-mistakes/SVM-12-03-18'

# Saving false positives as a matlab matrix for visualisation 
scio.savemat('false_negatives_svm.mat', mdict={'fn_mat': fn_mat})
# Saving false positives as a matlab matrix for visualisation 
scio.savemat('false_positives_svm.mat', mdict={'fp_mat': fp_mat})

# Saving false positives as a matlab matrix for visualisation 
scio.savemat('true_pos_easy.mat', mdict={'tp_easy_mat': tp_easy_mat})

# Saving false positives as a matlab matrix for visualisation 
scio.savemat('true_pos_hard.mat', mdict={'tp_hard_mat': tp_hard_mat})

# Saving false positives as a matlab matrix for visualisation 
scio.savemat('true_neg_easy.mat', mdict={'tn_easy_mat': tn_easy_mat})

# Saving false positives as a matlab matrix for visualisation 
scio.savemat('true_neg_hard.mat', mdict={'tn_hard_mat': tn_hard_mat})