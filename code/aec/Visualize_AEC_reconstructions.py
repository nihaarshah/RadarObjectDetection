decoded_imgs = autoencoder.predict(x_val)
idx_list = np.empty((0))
for j in range (len(y_val)):
	if y_val[j] ==1:
		idx_list.append[j] 

x_val_pos = x_val[idx_list,:,:]

n = 10
plt.figure(figsize=(20, 4))
for i in range(1,n):
    # display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_val[i+30].reshape(36, 36))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i+30].reshape(36, 36))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.ion()
plt.show()

n = 10
plt.figure(figsize=(20, 8))
for i in range(1,n):
    ax = plt.subplot(1, n, i)
    # plt.imshow(tf.transpose(tf.reshape(encoded[i],(5, 5 * 8))))
    plt.imshow(encoded_imgs[i].reshape(5,5*8).T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()