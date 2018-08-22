import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.svm import LinearSVC
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

#Path to positive features matlab array
positive_path1 = '/Users/nihaar/Documents/4yp/data/new-detections/taxi-rank-2/real_features_matrix.mat'
positive_path2 = '/Users/nihaar/Documents/4yp/data/new-detections/outside-uni-parks-1/real_features_matrix.mat'
positive_path3 = '/Users/nihaar/Documents/4yp/data/new-detections/nuffleld college 1/real_features_matrix.mat'
positive_path4 = '/Users/nihaar/Documents/4yp/data/new-detections/lamb and flag 1/real_features_matrix.mat'
positive_path5 = '/Users/nihaar/Documents/4yp/data/new-detections/broad street/real_features_matrix.mat'

negative_path1 = '/Users/nihaar/Documents/4yp/data/new-detections/taxi-rank-2/fake_features_matrix.mat'
negative_path2 = '/Users/nihaar/Documents/4yp/data/new-detections/outside-uni-parks-1/fake_features_matrix.mat'
negative_path3 = '/Users/nihaar/Documents/4yp/data/new-detections/nuffleld college 1/fake_features_matrix.mat'
negative_path4 = '/Users/nihaar/Documents/4yp/data/new-detections/lamb and flag 1/fake_features_matrix.mat'
negative_path5 = '/Users/nihaar/Documents/4yp/data/new-detections/broad street/fake_features_matrix.mat'

# Positive image samples 
pos_mat1 = scio.loadmat(positive_path1)
pp1 = pos_mat1['real_features_matrix']

pos_mat2 = scio.loadmat(positive_path2)
pp2 = pos_mat2['real_features_matrix']

pos_mat3 = scio.loadmat(positive_path3)
pp3 = pos_mat3['real_features_matrix']

pos_mat4 = scio.loadmat(positive_path4)
pp4 = pos_mat4['real_features_matrix']

pos_mat5 = scio.loadmat(positive_path5)
pp5 = pos_mat5['real_features_matrix']

# Negative images 
neg_mat1 = scio.loadmat(negative_path1)
np1 = neg_mat1['fake_features_matrix']

neg_mat2 = scio.loadmat(negative_path2)
np2 = neg_mat2['fake_features_matrix']

neg_mat3 = scio.loadmat(negative_path3)
np3 = neg_mat3['fake_features_matrix']

neg_mat4 = scio.loadmat(negative_path4)
np4 = neg_mat4['fake_features_matrix']

neg_mat5 = scio.loadmat(negative_path5)
np5 = neg_mat5['fake_features_matrix']

#Joining all real feature matrices from various datasets into one big matrix
positive_patch = np.concatenate((pp1,pp2,pp3,pp4,pp5),axis = 0)
negative_patch = np.concatenate((np1,np2,np3,np4,np5),axis =0)

# Now the generic indexing begins, agnostic of data matrices
i,_ = negative_patch.shape
j,_ = positive_patch.shape

# Split train and test data as 80%-20% of real and fake sets
i_train_boundary = np.ceil(0.85*i)
i_train_boundary = i_train_boundary.astype(int)

j_train_boundary = np.ceil(0.85*j)
j_train_boundary = j_train_boundary.astype(int)

# Splitting training and test set in positive examples
x_train_pos=positive_patch[0:j_train_boundary,:] 
x_test_pos=positive_patch[j_train_boundary+1:j,:] 

# Splitting training and test set in negative examples
x_train_neg=negative_patch[0:i_train_boundary,:] # Again leaving out 200 images;
x_test_neg = negative_patch[i_train_boundary+1:]

# Extracting the rows (i.e. no. of features) of all data matrices.
p,_ = np.shape(x_train_pos)
q,_ = np.shape(x_test_pos)
r,_ = np.shape(x_train_neg)
s,_ = np.shape(x_test_neg) 

# Now y stuff

# training labels for these images
y_train_pos = np.ones(p)
y_train_neg = np.zeros(r)
y_train = np.concatenate((y_train_pos,y_train_neg))
y_train = np.transpose(y_train)

# testing labels for the images
y_test_pos = np.ones(q)
y_test_neg = np.zeros(s)
y_test = np.concatenate((y_test_pos,y_test_neg))
y_test = np.transpose(y_test)


# Concatenating the positive and negative training and testing sets
x_train=np.concatenate((x_train_pos,x_train_neg),axis=0)
x_test=np.concatenate((x_test_pos,x_test_neg),axis=0)



# Randomly shuffling the rows of the feature matrix and corresponding target labels
y_train= np.reshape(y_train,[p+r,1])   
xy_combined = np.concatenate((x_train, y_train), axis=1)
np.random.shuffle(xy_combined) 
m,n = xy_combined.shape
y_train = xy_combined[:,n-1]
x_train = xy_combined[:,0:n-1]


# Same for test data
y_test = np.reshape(y_test,[q+s,1])
xy_combined = np.concatenate((x_test, y_test), axis=1)
np.random.shuffle(xy_combined) 
m,n = xy_combined.shape
y_test = xy_combined[:,n-1]
x_test = xy_combined[:,0:n-1]

#%%%%%%%%%%%%%%%%%%%%% END OF DATA PREPARATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Starting the SVM stuff

# Create a simple classifier
classifier = svm.LinearSVC()
classifier.fit(x_train, y_train)
y_score = classifier.decision_function(x_test)

# Print average pr value

average_precision = average_precision_score(y_test, y_score)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))

# Plot PR curve

precision, recall, _ = precision_recall_curve(y_test, y_score)

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