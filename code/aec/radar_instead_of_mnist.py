from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
import numpy as np
import scipy.io as scio
import tensorflow as tf
from keras.callbacks import TensorBoard

input_img = Input(shape=(28,28,1))

# x = Conv2D(8,(2,2), activation='relu',padding='same')(input_img)
# x = Conv2D(8,(2,2),activation='relu',padding='same')(x)
# encoded = MaxPooling2D((2,2),padding='same')(x)
# # at this point the representation is (3,3,8) i.e. 72 dim


x = Conv2D(16,(3,3), activation='relu',padding='same')(input_img)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(8,(3,3),activation='relu',padding='same')(x)
x = MaxPooling2D((2,2),padding='same')(x)
x = Conv2D(8,(3,3), activation='relu',padding='same')(x)
encoded = MaxPooling2D((2,2),padding='same')(x)

# # at this point the representation is (4,4,8) i.e. 128 dim
# x = UpSampling2D((2,2))(encoded)
# x = Conv2D(8,(2,2), activation='relu', padding='same')(x)
# x = Conv2D(16,(2,2),activation='relu',padding='same')(x)
# decoded = Conv2D(1,(2,2),activation='sigmoid',padding='same')(x)

x = Conv2D(8,(3,3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2,2))(x)
x = Conv2D(8,(3,3),activation='relu',padding='same')(x)
x = UpSampling2D((2,2))(x)
x = Conv2D(16,(3,3),activation='relu')(x)
x = UpSampling2D((2,2))(x)
decoded = Conv2D(1,(3,3),activation='sigmoid',padding='same')(x)

autoencoder = Model(input_img,decoded)
autoencoder.compile(optimizer='adadelta',loss = 'mean_squared_logarithmic_error')


mat = scio.loadmat('/Users/nihaar/Documents/4yp-code/data/TrainingTensorPedestrianNumpy.mat')
mat_array = mat['TrainingTensorPedestrianNumpy']
# x_train=mat_array[0:400,:,:]
# x_test=mat_array[401:500,:,:]


# x_train = x_train.astype('float32') /255.
# x_test = x_test.astype('float32')/255.
x_train = x_train.astype('float32') 
x_test = x_test.astype('float32')
x_train = np.reshape(x_train, (len(x_train), 28,28,1))
x_test = np.reshape(x_test, (len(x_test), 28,28,1))


autoencoder.fit(x_train,x_train,
                epochs=40,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test,x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder/')])

decoded_imgs = autoencoder.predict(x_test)

import matplotlib.pyplot as plt

n = 10
plt.figure(figsize=(20, 4))
for i in range(1,n):
    # display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

n = 10
plt.figure(figsize=(20, 8))
for i in range(1,n):
    ax = plt.subplot(1, n, i)
    plt.imshow(encoded[i].reshape(4, 4 * 8).T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

