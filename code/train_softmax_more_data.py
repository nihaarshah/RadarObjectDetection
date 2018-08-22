# This script loads the weights of unsupervised pretraining by AEC, pops the decoding layers
# of the AEC, appends softmax to it and trains on all the "new-detections". The dataset 
# trained on is the same as the SVM training dataset from 7th Jan 2018 so that we can 
# compare the results from each. The saved weights of the whole discriminator are then saved
# for future testing (using confusion matrix etc) in a different script.

# As of 9th Jan the test data and evaluation is happening in this script itself. Need to 
# figure out how to save the model weights and re-use later.

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
from sklearn.metrics import average_precision_score
from sklearn import datasets, svm, metrics

input_img = Input(shape=(36,36,1))

x = Conv2D(16,(3,3), activation='relu',padding='same')(input_img)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(8,(3,3),activation='relu',padding='same')(x)
x = MaxPooling2D((2,2),padding='same')(x)
x = Conv2D(8,(3,3), activation='relu',padding='same')(x)
encoded = MaxPooling2D((2,2),padding='same')(x)

# at this point the representation is (4,4,8) i.e. 128 dim

x = Conv2D(8,(3,3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2,2))(x)
x = Conv2D(8,(3,3),activation='relu',padding='same')(x)
x = UpSampling2D((2,2))(x)
x = Conv2D(16,(3,3),activation='relu')(x)
x = UpSampling2D((2,2))(x)
decoded = Conv2D(1,(3,3),activation='relu',padding='same')(x)

# Initialize the autoencoder model, note it comes with randomly pre-initialised weights that can
# be viewed using autoencoder.get_weights() in ipython
autoencoder = Model(input_img,decoded)

# Load the training weights from the AEC model into this model 
autoencoder.load_weights('/Users/nihaar/Documents/4yp/4yp/code/aec/autoencoder_model_weights3.h5', by_name=True)
print('Loaded pretrained weights from autoencoder model')

# Truncating the layers of the decoder so we are only left with the encoder
for i in range (1,7):
    autoencoder.layers.pop()

# Add layer to reshape the encoded tensor as a flattened 2D matrix of size [N_images x N_features] where N_feat= 5x5x8
pool2_flat = Flatten()
inp = autoencoder.input
out = pool2_flat(autoencoder.layers[-1].output)
classifier = Model(inp, out)

# Add a dense softmax on top of the reshaped layer
fc_layer = Dense(200, activation='sigmoid', name='my_dense')
inp = classifier.input
out = fc_layer(classifier.layers[-1].output)
classifier = Model(inp, out)

fc_layer = Dense(1, activation='sigmoid', name='my_2nd_dense')
inp = classifier.input
out = fc_layer(classifier.layers[-1].output)
classifier = Model(inp, out)

# Freeze lower 2 layers
# for layer in autoencoder.layers[:2]:
#     layer.trainable = False

# Create the final classifier model and display
classifier = Model(inp, out)
classifier.summary(line_length=150)


# Prepare the data
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
positive_patch = np.concatenate((pp1,pp2,pp3,pp5),axis = 0)
negative_patch = np.concatenate((np1,np3),axis =0)

# Preparing the test data
pp_test = pp4
np_test = np4
x,_ = pp_test.shape
y,_ = np_test.shape
pp_tens_test = np.reshape(pp_test,[x,41,41])
np_tens_test = np.reshape(np_test,[y,41,41])
# Trimming down the patches
pp_tens_test = pp_tens_test[:,2:38,2:38]
np_tens_test = np_tens_test[:,2:38,2:38]
x_test = np.concatenate((pp_tens_test,np_tens_test),axis =0)
x_test = x_test.astype('float32')
x_test = np.reshape(x_test, (len(x_test), 36,36,1))
y_test_pos = np.ones(x)
y_test_neg = np.zeros(y)
y_test = np.concatenate((y_test_pos,y_test_neg))

# Test data prepared
# Preparing training data
m,n = positive_patch.shape
i,j = negative_patch.shape

# Trim 41 x 41 dimension to 36 x 36 to suit architecture
pp_tens = np.reshape(positive_patch,[m,41,41])
np_tens = np.reshape(negative_patch,[i,41,41])
# Trimming down the patches
pp_tens = pp_tens[:,2:38,2:38]
np_tens = np_tens[:,2:38,2:38]

pp_np_tens = np.concatenate((pp_tens,np_tens),axis =0)
y_train_pos = np.ones(m)
y_train_neg = np.zeros(i)
y = np.concatenate((y_train_pos,y_train_neg))

#Splitting training data randomly for the purpose of validation (NB there is no testing happening in this script)
X_train, X_val, y_train, y_val = train_test_split(pp_np_tens, y, test_size=0.33, random_state=42)


X_train = X_train.astype('float32') 
X_val = X_val.astype('float32')
X_train = np.reshape(X_train, (len(X_train), 36,36,1))
X_val = np.reshape(X_val, (len(X_val), 36,36,1))

classifier.compile(optimizer='adadelta',loss = 'binary_crossentropy', metrics=['accuracy'])
classifier.fit(X_train,y_train,
                epochs=50,
                verbose=2,
                batch_size=128,
                shuffle=True,
                validation_data=(X_val,y_val),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder/')])


# Evaluating on test data
# Test prediction probabilities
y_hat = classifier.predict(x_test)
y_hat = y_hat[:] > 0.5 # converts predictions to binary format
# Print average pr value
average_precision = average_precision_score(y_test, y_hat)
print('Average precision-recall score: {0:0.2f}'.format(average_precision))


print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, y_hat))

classifier.metrics_names

classifier.save('/Users/nihaar/Documents/4yp/data/weights/25-02-18/softmax_model_with_pretraining_without_greedy_pretraining.h5')
print("Saved model to disk")


# loss,acc = classifier.evaluate(x_test, y_test, verbose=1)
# print(loss, acc)
