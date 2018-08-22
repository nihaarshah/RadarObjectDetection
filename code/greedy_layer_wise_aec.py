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
import load_data_rotated 
from sklearn.model_selection import train_test_split

# Data prep
x,y = load_data_rotated.load_train_data_AEC_rotated()
x = x.astype('float32') 
#Splitting training data randomly for the purpose of validation (NB there is no testing happening in this script)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.05, random_state=42)

x_train = x_train.astype('float32') 
x_val = x_val.astype('float32')
x_train = np.reshape(x_train, (len(x_train), 36,36,1))
x_val = np.reshape(x_val, (len(x_val), 36,36,1))


# AEC begins
input_img = Input(shape=(36,36,1))

#Layer one training

x = Conv2D(16,(3,3), activation='relu',padding='same')(input_img)
encoded = MaxPooling2D((2,2), padding='same')(x)
x = UpSampling2D((2,2))(encoded)
decoded = Conv2D(1,(3,3),activation='relu',padding='same')(x)

autoencoder = Model(input_img,decoded)

autoencoder.compile(optimizer='adadelta',loss = 'mean_squared_error')

autoencoder.fit(x_train,x_train,
                epochs=5,
                batch_size=128,
                shuffle=True,
                validation_data=(x_val,x_val),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder_greedy1/')])

autoencoder.save_weights('autoencoder_model_weights1.h5')

# Layer 2
x = Conv2D(16,(3,3), activation='relu',padding='same')(input_img)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(8,(3,3),activation='relu',padding='same')(x)
encoded = MaxPooling2D((2,2),padding='same')(x)

x = UpSampling2D((2,2))(encoded)
x = Conv2D(16,(3,3),activation='relu',padding='same')(x)
x = UpSampling2D((2,2))(x)
decoded = Conv2D(1,(3,3),activation='relu',padding='same')(x)

autoencoder = Model(input_img,decoded)

autoencoder.load_weights('/Users/nihaar/Documents/4yp/4yp/code/aec/autoencoder_model_weights1.h5', by_name=True)
print('Loaded pretrained weights from autoencoder model')

freeze1 = autoencoder.layers[1]
freeze1.trainable = False
freeze2 = autoencoder.layers[8]
freeze2.trainable = False
autoencoder.compile(optimizer='adadelta',loss = 'mean_squared_error')

autoencoder.fit(x_train,x_train,
                epochs=10,
                batch_size=128,
                shuffle=True,
                validation_data=(x_val,x_val),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder_greedy2/')])

autoencoder.save_weights('autoencoder_model_weights2.h5')


# Layer 3
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

autoencoder = Model(input_img,decoded)

autoencoder.load_weights('/Users/nihaar/Documents/4yp/4yp/code/aec/autoencoder_model_weights2.h5', by_name=True)
print('Loaded pretrained weights from autoencoder model')

freeze1 = autoencoder.layers[1]
freeze1.trainable = False
freeze2 = autoencoder.layers[3]
freeze2.trainable = False
freeze3 = autoencoder.layers[11]
freeze3.trainable = False
freeze4 = autoencoder.layers[13]
freeze4.trainable = False

autoencoder.compile(optimizer='adadelta',loss = 'mean_squared_error')

autoencoder.fit(x_train,x_train,
                epochs=10,
                batch_size=128,
                shuffle=True,
                validation_data=(x_val,x_val),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder_greedy3')])

autoencoder.save_weights('autoencoder_model_weights3.h5')