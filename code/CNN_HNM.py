# Training which involves hard negative mining on the validation set
import scipy.io as scio
import numpy as np
from sklearn import datasets, svm, metrics
from sklearn.svm import LinearSVC
import pickle
from sklearn.model_selection import train_test_split
import sys
sys.path.insert(0, '/Users/nihaar/Documents/4yp/4yp/code/helpers/')
import load_CNN_data_aligned 
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

x_train,y_train= load_CNN_data_aligned.load_train_data_DNN()

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.33, random_state=42)

x_test,y_test = load_CNN_data_aligned.load_test_data_DNN()

# Starting the SVM stuff
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.svm import LinearSVC

# Create a classifier: a support vector classifier
classifier = svm.SVC(C=.05,kernel='linear',probability=True)
# Creating a file to which we will print the results
f = open('svm_hard_neg_min_results_12march.txt','w')
f.write('SVM Hard negative mining report:')
f.close()
classifier.fit(x_train, y_train)
for rounds in range(0,4):
  print("Round started # %s"%(rounds))
  # Using the validation data to evaluate and later do HNM
  y_score = classifier.decision_function(x_val)
  y_proba = classifier.predict_proba(x_val)
  i = len(y_val)
  fp = []
  # Vector of indices of false-positives in the validation set
  for j in range(0,i):
    if y_val[j] == 0 and y_score[j] >= 0:
      if 0.60 < y_proba[j,1] < 0.70:
          fp.append(j)

  # Create a matrix of false negative features and corresponding labels
  hn_mat = np.empty([0,1681])
  hn_mat = x_val[fp,:]
  y_hn = np.zeros(len(fp))

  # Add the fp and fn to the training set
  x_train = np.concatenate((x_train,hn_mat),axis=0)
  y_train = np.concatenate((y_train,y_hn))

  # Retrain
  classifier.fit(x_train, y_train)  

  # Test on the test set
  predicted = classifier.predict(x_test)

  # Calculate metrics
  expected = y_test
  average_precision = average_precision_score(expected, predicted)

  # Write results for each round to a file
  f = open('svm_hard_neg_min_results_12march.txt','a')
  f.write('Round # %s\n'%(rounds))
  f.write('Number of false positives: %s\n'%(len(fp)))
  f.write('\n'+'Average precision-recall score: {0:0.2f}'.format(
        average_precision))
  f.write('\n'+ "Classification report for classifier %s:\n%s\n"% (classifier, metrics.classification_report(expected, predicted)))
  f.write('\n'+ "Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
  f.close()
  filename = 'svm_hard_neg_mined_cross_val_model_rotated%d.pkl'%(rounds)
  pickle.dump(classifier, open(filename, 'wb'))

# # Visualising False negatives
# import matplotlib.pyplot as plt

# n = 10
# plt.figure(figsize=(20, 4))
# for i in range(1,n):
#     # display original
#     ax = plt.subplot(2, n, i)
#     plt.imshow(fp_mat[i,:].reshape(41, 41))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)

# plt.show()
# Saving the final model to plot pr curve later
# Save the model weights
filename = 'svm_hard_neg_mined_cross_val_model_rotated.pkl'
pickle.dump(classifier, open(filename, 'wb'))

# Saving false positives as a matlab matrix for visualisation 
scio.savemat('false_negatives_svm.mat', mdict={'hn_mat': hn_mat})
