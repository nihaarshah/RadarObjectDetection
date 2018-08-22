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
import sys
sys.path.insert(0, '/Users/nihaar/Documents/4yp/4yp/code/helpers/')
import load_data_rotated 

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
decoded = Conv2D(1,(3,3),activation='sigmoid',padding='same')(x)

# Initialize the autoencoder model, note it comes with randomly pre-initialised weights that can
# be viewed using autoencoder.get_weights() in ipython
autoencoder = Model(input_img,decoded)

# Load the training weights from the AEC model into this model 
autoencoder.load_weights('/Users/nihaar/Documents/4yp/data/weights/autoencoder_model_weights.h5', by_name=True)
print('Loaded pretrained weights from autoencoder model')

# Truncating the layers of the decoder so we are only left with the encoder
for i in range (1,8):
    autoencoder.layers.pop()

# Add layer to reshape the encoded tensor as a flattened 2D matrix of size [N_images x N_features] where N_feat= 5x5x8
pool2_flat = Flatten()
inp = autoencoder.input
out = pool2_flat(autoencoder.layers[-1].output)
classifier = Model(inp, out)

# Add a dense on top of the reshaped layer
fc_layer1 = Dense(200, name='my_dense1')
inp = autoencoder.input
out = fc_layer1(classifier.layers[-1].output)
classifier = Model(inp, out)

# Add a aother dense on top of the reshaped layer
fc_layer2 = Dense(1, name='my_dense2')
inp = autoencoder.input
out = fc_layer2(classifier.layers[-1].output)

# Create the final classifier model and display
classifier = Model(inp, out)
classifier.summary(line_length=150)

# Prepare the data
x_train,y_train = load_data_rotated.load_train_data_DNN_rotated_regression_train()
x_test,y_test = load_data_rotated.load_train_data_DNN_rotated_regression_test()

#Splitting training data randomly for the purpose of validation (NB there is no testing happening in this script)
X_train, X_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.15, random_state=42)

X_train = X_train.astype('float32') 
X_val = X_val.astype('float32')
X_train = np.reshape(X_train, (len(X_train), 36,36,1))
X_val = np.reshape(X_val, (len(X_val), 36,36,1))

classifier.compile(optimizer='adadelta',loss = 'mean_squared_error', metrics=['mse'])
classifier.fit(X_train,y_train,
                epochs=50,
                verbose=2,
                batch_size=128,
                shuffle=True,
                validation_data=(X_val,y_val),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder/')])


classifier.save('/Users/nihaar/Documents/4yp/data/weights/15-02-18/angle_regression_taxi_2dense.h5')
print("Saved model to disk")

