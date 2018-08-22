from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
import numpy as np
import scipy.io as scio
import tensorflow as tf
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from random import shuffle
import sys
sys.path.insert(0, '/Users/nihaar/Documents/4yp/4yp/code/helpers/')
import load_CNN_data_aligned 
# import load_data_rotated 
from sklearn.model_selection import train_test_split

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

x,y = load_CNN_data_aligned.load_train_data_DNN()
x = x.astype('float32') 
#Splitting training data randomly for the purpose of validation (NB there is no testing happening in this script)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.05, random_state=42)

x_train = x_train.astype('float32') 
x_val = x_val.astype('float32')
x_train = np.reshape(x_train, (len(x_train), 36,36,1))
x_val = np.reshape(x_val, (len(x_val), 36,36,1))

# Create the final classifier model and display
# classifier = Model(input_img, decoded)
autoencoder.summary(line_length=150)

autoencoder.fit(x_train,x_train,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_data=(x_val,x_val),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder/')])
#Saving the weights
autoencoder.save_weights('autoencoder_model_weights.h5')

# decoded_imgs = autoencoder.predict(x_val)



# n = 10
# plt.figure(figsize=(20, 4))
# for i in range(1,n):
#     # display original
#     ax = plt.subplot(2, n, i)
#     plt.imshow(x_test[i+30].reshape(36, 36))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#     # display reconstruction
#     ax = plt.subplot(2, n, i + n)
#     plt.imshow(decoded_imgs[i+30].reshape(36, 36))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)

# plt.show()

# n = 10
# plt.figure(figsize=(20, 8))
# for i in range(1,n):
#     ax = plt.subplot(1, n, i)
#     plt.imshow(tf.transpose(tf.reshape(encoded[i],(5, 5 * 8))))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()

