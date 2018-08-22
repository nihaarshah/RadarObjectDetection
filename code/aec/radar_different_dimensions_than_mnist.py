from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
import numpy as np
import scipy.io as scio
import tensorflow as tf
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from random import shuffle

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

autoencoder = Model(input_img,decoded)
autoencoder.compile(optimizer='adadelta',loss = 'mean_squared_error')


pos_mat = scio.loadmat('/Users/nihaar/Documents/4yp-code/data/taxi-rank2-car-patches/XTrainTrimmedPosThresholded.mat')
positive_patch = pos_mat['XTrainTrimmedPosThresholded']

neg_mat = scio.loadmat('/Users/nihaar/Documents/4yp-code/data/taxi-rank2-car-patches/XTrainTrimmedNegThresholded.mat')
negative_patch = neg_mat['XTrainTrimmedNegThresholded']

x_train_pos=positive_patch[0:5120,:,:] #Leaving out 10 images; 100 for testing AEC; maybe 100 for testing softmax later
x_test_pos=positive_patch[5121:5131,:,:] # 10 should suffice to "test"

x_train_neg=negative_patch[0:4059,:,:] # Again leaving out 10 images;
x_test_neg = negative_patch[4060:4070,:,:]

# Concatenating the positive and negative training and "testing" sets
x_train=np.concatenate((x_train_neg,x_train_pos),axis=0)
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


# x_train = x_train.astype('float32') /255.
# x_test = x_test.astype('float32')/255.
x_train = x_train.astype('float32') 
x_test = x_test.astype('float32')
x_train = np.reshape(x_train, (len(x_train), 36,36,1))
x_test = np.reshape(x_test, (len(x_test), 36,36,1))


autoencoder.fit(x_train,x_train,
                epochs=40,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test,x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder/')])
#Saving the weights
autoencoder.save_weights('autoencoder_model_weights.h5')

decoded_imgs = autoencoder.predict(x_test)



n = 10
plt.figure(figsize=(20, 4))
for i in range(1,n):
    # display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i+30].reshape(36, 36))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i+30].reshape(36, 36))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

n = 10
plt.figure(figsize=(20, 8))
for i in range(1,n):
    ax = plt.subplot(1, n, i)
    plt.imshow(tf.transpose(tf.reshape(encoded[i],(5, 5 * 8))))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

