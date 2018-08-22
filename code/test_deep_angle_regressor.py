# To test the regressor to predict angles
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
from keras.models import load_model
from sklearn import datasets, svm, metrics
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
import sys
sys.path.insert(0, '/Users/nihaar/Documents/4yp/4yp/code/helpers/')
import load_data_rotated 

x_test,y_test = load_data_rotated.load_train_data_DNN_rotated_regression_test()

# Load model
classifier = load_model('/Users/nihaar/Documents/4yp/data/weights/15-02-18/angle_regression_taxi.h5')

x_train
x_test
x_test = x_test.astype('float32')
x_test = np.reshape(x_test, (len(x_test), 36,36,1))


#Predict scores
y_hat = classifier.predict(x_test)