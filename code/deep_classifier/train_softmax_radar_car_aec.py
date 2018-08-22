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

# Add a dense softmax on top of the reshaped layer
fc_layer = Dense(2, activation='sigmoid', name='my_dense')
inp = autoencoder.input
out = fc_layer(classifier.layers[-1].output)

# Create the final classifier model and display
classifier = Model(inp, out)
classifier.summary(line_length=150)


# Do the X stuff

# Positive image samples 
pos_mat = scio.loadmat('/Users/nihaar/Documents/4yp/data/misc/taxi-rank2-car-patches/XTrainTrimmedPosThresholded.mat')
positive_patch = pos_mat['XTrainTrimmedPosThresholded']

# Negative images 
neg_mat = scio.loadmat('/Users/nihaar/Documents/4yp/data/misc/taxi-rank2-car-patches/XTrainTrimmedNegThresholded.mat')
negative_patch = neg_mat['XTrainTrimmedNegThresholded']

# Splitting training and test set in positive examples
x_train_pos=positive_patch[0:4360,:,:] 
x_test_pos=positive_patch[4360:5132,:,:] 

# Splitting training and test set in negative examples
x_train_neg=negative_patch[0:3460,:,:] # Again leaving out 200 images;
x_test_neg = negative_patch[3460:4071,:,:]

# Now y stuff

# training labels for these images
y_train_mat = scio.loadmat('/Users/nihaar/Documents/4yp/data/misc/taxi-rank2-car-patches/yTraining.mat')
y_train = y_train_mat['yTraining']
y_train = np.transpose(y_train)

# testing labels for the images
y_test_mat = scio.loadmat('/Users/nihaar/Documents/4yp/data/misc/taxi-rank2-car-patches/yTest.mat')
y_test = y_test_mat['yTest']
y_test = np.transpose(y_test)


# Concatenating the positive and negative training and testing sets
x_train=np.concatenate((x_train_pos,x_train_neg),axis=0)
x_test=np.concatenate((x_test_pos,x_test_neg),axis=0)

# This block doesnt work; find out why; would be nice to have shuffled images
# # Shuffle the indices of the stacked images so that the pos and neg dont appear in a block
# ind_list = [i for i in range(x_train.shape[0])]
# shuffle(ind_list)
# x_train  = x_train[ind_list,:,:,:]
# # Same thing for the test
# ind_list = [i for i in range(x_test.shape[0])]
# shuffle(ind_list)
# x_test = x_test[ind_list,:,:,:]


x_train = x_train.astype('float32') 
x_test = x_test.astype('float32')
# Why are we adding a 4th dimension
x_train = np.reshape(x_train, (len(x_train), 36,36,1))
x_test = np.reshape(x_test, (len(x_test), 36,36,1))

classifier.compile(optimizer='adadelta',loss = 'mean_squared_error', metrics=['accuracy'])
classifier.fit(x_train,y_train,
                epochs=2,
                verbose=2,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test,y_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder/')])

# y_hat = classifier.predict(x_test)

# loss,acc = classifier.evaluate(x_test, y_test, verbose=1)
# print(loss, acc)

classifier.metrics_names

classifier.save('/Users/nihaar/Documents/4yp/data/autoencoder.h5')
print("Saved model to disk")



