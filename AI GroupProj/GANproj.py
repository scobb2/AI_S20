# for data input and output:
import numpy as np
from numpy.random import rand, randn, randint
from numpy import ones, zeros, vstack
import os

# for deep learning: 
import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras import models, layers
from tensorflow.keras.layers import Dense, Conv2D, Dropout # Input
from tensorflow.keras.layers import Flatten # BatchNormalization
from tensorflow.keras.layers import Activation, LeakyReLU
from tensorflow.keras.layers import Reshape # new!
from tensorflow.keras.layers import Conv2DTranspose 
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.utils.data_utils import Sequence

# for plotting images:
from matplotlib import pyplot as plt

# Load data from mnist
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()
print(train_data.shape)
print(test_data.shape)

# Reshape training and test data to add an additional dimension of 1 channel
train_data = train_data.reshape((60000, 28, 28, 1))
test_data = test_data.reshape((10000, 28, 28, 1))

train_data = train_data[:60000]   # range indexing capability (take elements 0 through 59999)
test_data = test_data[:10000]

# Revise pixel data to 0.0 to 1.0, 32-bit float (this isn't quantum science)
train_data = train_data.astype('float32') / 255  # ndarray/scalar op
test_data = test_data.astype('float32') / 255


# Turn 1-value labels to 10-value vectors
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

train_labels = train_labels[:60000]     # range indexing capability (take elements 0 through 59999)
test_labels = test_labels[:10000]

# creating the descriminator
def build_discriminator(in_shape = (28,28,1)):
	dscrm = models.Sequential()
	dscrm.add(Conv2D(64, (3,3), strides = (2, 2), padding = 'same', input_shape = in_shape))
	dscrm.add(LeakyReLU(alpha = 0.2))
	dscrm.add(Dropout(0.4))
	dscrm.add(Conv2D(64, (3,3), strides = (2, 2), padding = 'same'))
	dscrm.add(LeakyReLU(alpha = 0.2)) # alpha = 0.2?, default is 0.3
	dscrm.add(Dropout(0.4))
	dscrm.add(Flatten())
	dscrm.add(Dense(1, activation = 'sigmoid'))
	dscrm.compile(loss = 'binary_crossentropy', optimizer = Nadam(lr = 0.0002, beta_1 = 0.5), metrics = ['accuracy'])
	return dscrm

# create batches of real samples labeled as real
def generate_real_samples(data, n_samples):
	randImg = randint(0, data.shape[0], n_samples) # choose random instances
	x = data[randImg] # retrieve selected images
	y = ones((n_samples, 1)) # fill labels with 1s to denote that they are real
	return x, y #image, label
 
# # generate n fake samples with corresponding labels
# def generate_fake_samples(n_samples):
# 	x = rand(28 * 28 * n_samples) # generate uniform random numbers in [0,1]
# 	x = x.reshape((n_samples, 28, 28, 1)) # reshape into a batch of n_samples size
# 	y = zeros((n_samples, 1)) # fill labels with 0s to denote that they are fake
# 	return x, y # fake image, label
 
# training the discriminator
def train_discriminator(Dnet, data, n_iter=100, n_batch=256):
	halfBatch = int(n_batch / 2) # used to create half real half fake dataset to train on
	for i in range(n_iter):
		x_real, y_real = generate_real_samples(data, halfBatch) # get randomly selected 'real' samples
		_, real_acc = Dnet.train_on_batch(x_real, y_real) # train discriminator on real samples
		x_fake, y_fake = generate_fake_samples(halfBatch) # generate 'fake' examples
		_, fake_acc = Dnet.train_on_batch(x_fake, y_fake) # train discriminator on fake samples
		print('>%d real=%.0f%% fake=%.0f%%' % (i+1, real_acc*100, fake_acc*100)) # display accuracy values
 
# # prepping and displaying discriminator
# discriminator = build_discriminator()
# discriminator.summary()

# # fit the discriminator model
# train_discriminator(discriminator, train_data)

# creating the generator
def build_generator(latent_dim):
	gen = models.Sequential()
	n_nodes = 128 * 7 * 7 # foundation for 7x7 image
	gen.add(Dense(n_nodes, input_dim = latent_dim))
	gen.add(LeakyReLU(alpha = 0.2))
	gen.add(Reshape((7, 7, 128)))
	gen.add(Conv2DTranspose(128, (4,4), strides = (2,2), padding = 'same')) # upscale to 14x14, transpose is similar to upSampling2D
	gen.add(LeakyReLU(alpha = 0.2))
	gen.add(Conv2DTranspose(128, (4,4), strides = (2,2), padding = 'same')) # upscale to 28x28
	gen.add(LeakyReLU(alpha = 0.2))
	gen.add(Conv2D(1, (7,7), activation = 'sigmoid', padding = 'same'))
	return gen
 
# generate random points in latent space to input into the generator
def generate_latent_points(latent_dim, n_samples):
	x_input = randn(latent_dim * n_samples) # generate rand points in the latent space
	x_input = x_input.reshape(n_samples, latent_dim) # reshape inputs for the network
	return x_input
 
# use the generator to generate n fake examples with corresponding labels
def generate_fake_samples(generator, latent_dim, n_samples):
	x_input = generate_latent_points(latent_dim, n_samples) # generate rand points in latent space
	x = generator.predict(x_input) # predict outputs
	y = zeros((n_samples, 1)) # create 'fake' class labels (0)
	return x, y
 
# # size of the latent space
# latent_dim = 100
# # define the discriminator model
# generator = build_generator(latent_dim)
# # generate samples
# n_samples = 25
# X, _ = generate_fake_samples(generator, latent_dim, n_samples)
# # plot the generated samples
# for i in range(n_samples):
# 	# define subplot
# 	plt.subplot(5, 5, 1 + i)
# 	# turn off axis labels
# 	plt.axis('off')
# 	# plot single image
# 	plt.imshow(X[i, :, :, 0], cmap = 'gray_r')
# # show the figure
# plt.show()

# creating the GAN
def build_gan(Gnet, Dnet):
	Dnet.trainable = False # "freeze" weights in the discriminator so that it isn't training

	gan = models.Sequential()
	
	gan.add(Gnet) # add the generator
	gan.add(Dnet) # add the discriminator

	gan.compile(loss = 'binary_crossentropy', optimizer = Nadam(lr = 0.0002, beta_1 = 0.5))
	return gan

# create and save generated images every 10 epochs
def save_plot(examples, epoch, n = 10):
	for i in range(n * n):
		plt.subplot(n, n, 1 + i)
		plt.axis('off')
		plt.imshow(examples[i, :, :, 0], cmap = 'gray_r') # plot raw pixel data and inverse the colors
	# save plots to file
	filename = 'generated_plot_e%03d.png' % (epoch+1)
	plt.savefig(filename)
	plt.close()

# evaluate the discriminator, plot the generated images, and save the generator model
def summarize_performance(epoch, Gnet, Dnet, data, latent_dim, n_samples = 100):
	x_real, y_real = generate_real_samples(data, n_samples) # prepare real samples
	_, acc_real = Dnet.evaluate(x_real, y_real, verbose = 0) # evaluate discriminator on real examples
	x_fake, y_fake = generate_fake_samples(Gnet, latent_dim, n_samples) # prepare fake examples
	_, acc_fake = Dnet.evaluate(x_fake, y_fake, verbose = 0) # evaluate discriminator on fake examples
	print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100)) # summarize discriminator accuracy
	save_plot(x_fake, epoch) # save plots
    # save the generator model
	filename = 'generator_model_%03d.h5' % (epoch + 1)
	Gnet.save(filename)
 
# train the generator and discriminator
def train(Gnet, Dnet, gan, data, latent_dim, n_epochs = 100, n_batch = 256):
	bat_per_epo = int(data.shape[0] / n_batch)
	half_batch = int(n_batch / 2)
	for i in range(n_epochs):
		for j in range(bat_per_epo):
			x_real, y_real = generate_real_samples(data, half_batch) # get randomly selected real samples
			x_fake, y_fake = generate_fake_samples(Gnet, latent_dim, half_batch) # generate fake samples
            # np vstack takes 2 ndarrays and makes an ndarray 1-D greater with both
			x, y = vstack((x_real, x_fake)), vstack((y_real, y_fake)) # create training set for the discriminator
			d_loss, _ = Dnet.train_on_batch(x, y) # update discriminator model weights
			x_gan = generate_latent_points(latent_dim, n_batch) # prepare points in latent space as input for the generator
			y_gan = ones((n_batch, 1)) # label the fake samples as real to fool the discriminator
			g_loss = gan.train_on_batch(x_gan, y_gan) # update the generator based on the discriminator's error
            # loss should stay pretty constant throughout the training processes
			print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, bat_per_epo, d_loss, g_loss)) # summarize loss on the batch
        # evaluate the model performance every 10 epochs
		if (i+1) % 10 == 0:
			summarize_performance(i, Gnet, Dnet, data, latent_dim)
 
latent_dim = 100 # size of the latent space

discriminator = build_discriminator() # create the discriminator

generator = build_generator(latent_dim) # create the generator

gan = build_gan(generator, discriminator) # create the gan

train(generator, discriminator, gan, train_data, latent_dim) # train it all together