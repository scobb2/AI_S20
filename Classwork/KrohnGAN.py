# From GAN example in Krohn, Deep Learning Illustrated

# for data input and output:
import numpy as np
import os

# for deep learning: 
import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Dropout
from tensorflow.keras.layers import BatchNormalization, Flatten
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Reshape # new! 
from tensorflow.keras.layers import Conv2DTranspose, UpSampling2D # new! 
from tensorflow.keras.optimizers import Nadam 

# for plotting: 
import pandas as pd
from matplotlib import pyplot as plt

input_images = "apple.npy"
data = np.load(input_images)  # shape (.., 782)
print(data.shape)

data = data/255
data = np.reshape(data,(data.shape[0], 28, 28, 1))

plt.imshow(data[4242,:,:,0], cmap='Greys')

def build_discriminator(depth=64, p=0.4):
    # Define inputs
    image = Input(data.shape[1:])  # All but the batch dimension
    
    # Convolutional layers
    conv1 = Dropout(p)(Conv2D(depth, 5, strides=2, padding='same', #stride by 2 to reduce the size instead of maxPooling
        activation='relu')(image))
    
    conv2 = Dropout(p)(Conv2D(depth*2, 5, strides=2, padding='same',
        activation='relu')(conv1))
    
    conv3 = Dropout(p)(Conv2D(depth*4, 5, strides=2, padding='same',
        activation='relu')(conv2))

    conv4 = Dropout(p)(Conv2D(depth*8, 5, strides=1, padding='same',
        activation='relu')(conv3))
    
    # Output layer
    prediction = Dense(1, activation='sigmoid')(Flatten()(conv4))
    
    # Model definition
    model = Model(inputs=image, outputs=prediction)
    
    return model

discriminator = build_discriminator()
discriminator.summary()

discriminator.compile(loss='binary_crossentropy', optimizer=Nadam(lr=0.0008, 
    clipvalue=1.0), metrics=['accuracy'])

z_dimensions = 32

def build_generator(latent_dim=z_dimensions, depth=64, p=0.4):
    noise = Input((latent_dim,))
    
    # First dense layer
    dense1 = Dense(7*7*depth)(noise)
    dense1 = BatchNormalization(momentum=0.9)(dense1)
    dense1 = Activation(activation='relu')(dense1)
    dense1 = Reshape((7,7,depth))(dense1)
    dense1 = Dropout(p)(dense1)
    
    # De-Convolutional layers
    conv1 = UpSampling2D()(dense1)
    conv1 = Conv2DTranspose(int(depth/2), kernel_size=5, padding='same', 
        activation=None,)(conv1)
    conv1 = BatchNormalization(momentum=0.9)(conv1)
    conv1 = Activation(activation='relu')(conv1)
    
    conv2 = UpSampling2D()(conv1)
    conv2 = Conv2DTranspose(int(depth/4), kernel_size=5, padding='same', 
        activation=None,)(conv2)
    conv2 = BatchNormalization(momentum=0.9)(conv2)
    conv2 = Activation(activation='relu')(conv2)
    
    conv3 = Conv2DTranspose(int(depth/8), kernel_size=5, padding='same', 
        activation=None,)(conv2)
    conv3 = BatchNormalization(momentum=0.9)(conv3)
    conv3 = Activation(activation='relu')(conv3) # 28 x 28 output at this point

    # Output layer
    image = Conv2D(1, kernel_size=5, padding='same', activation='sigmoid')(conv3)

    # Model definition    
    model = Model(inputs=noise, outputs=image)
    
    return model

generator = build_generator()
generator.summary()

z = Input(shape=(z_dimensions,))
img = generator(z)
discriminator.trainable = False
pred = discriminator(img)

adversarial_model = Model(z, pred)
adversarial_model.compile(loss='binary_crossentropy', 
    optimizer=Nadam(lr=0.0004, clipvalue=1.0), metrics=['accuracy'])

def train(epochs=2000, batch=128, z_dim=z_dimensions):    
    d_metrics = []
    a_metrics = []
    
    running_d_loss = 0
    running_d_acc = 0
    running_a_loss = 0
    running_a_acc = 0
    
    for i in range(epochs):
        
        # sample real images:
        random_pick = np.random.choice(data.shape[0], batch, replace=False) 
        real_imgs = np.reshape(data[random_pick], (batch,28,28,1))
        
        # generate fake images: 
        fake_imgs = generator.predict(
            np.random.uniform(-1.0, 1.0, size=[batch, z_dim]))
        
        # concatenate images as discriminator inputs:
        x = np.concatenate((real_imgs,fake_imgs))
        
        # assign y labels for discriminator: 
        y = np.ones([2*batch,1])
        y[batch:,:] = 0
        
        # train discriminator: 
        d_metrics.append(
            discriminator.train_on_batch(x,y)
        )
        running_d_loss += d_metrics[-1][0]
        running_d_acc += d_metrics[-1][1]
        
        # adversarial net's noise input and "real" y: 
        noise = np.random.uniform(-1.0, 1.0, 
                                  size=[batch, z_dim])
        y = np.ones([batch,1])
        
        # train adversarial net: 
        a_metrics.append(
            adversarial_model.train_on_batch(noise,y)
        ) 
        running_a_loss += a_metrics[-1][0]
        running_a_acc += a_metrics[-1][1]
        
        # periodically print progress & fake images: 
        if (i+1)%100 == 0:
            print('Epoch #{}'.format(i))
            log_mesg = "%d: [D loss: %f, acc: %f]" % \
            (i, running_d_loss/i, running_d_acc/i)
            log_mesg = "%s  [A loss: %f, acc: %f]" % \
            (log_mesg, running_a_loss/i, running_a_acc/i)
            print(log_mesg)

            noise = np.random.uniform(-1.0, 1.0, 
                                      size=[16, z_dim])
            gen_imgs = generator.predict(noise)

            plt.figure(figsize=(5,5))

            for k in range(gen_imgs.shape[0]):
                plt.subplot(4, 4, k+1)
                plt.imshow(gen_imgs[k, :, :, 0], 
                           cmap='gray')
                plt.axis('off')
                
            plt.tight_layout()
            plt.show()
    
    return a_metrics, d_metrics

a_metrics_complete, d_metrics_complete = train()

ax = pd.DataFrame(
    {
        'Adversarial': [metric[0] for metric in a_metrics_complete],
        'Discriminator': [metric[0] for metric in d_metrics_complete],
    }
).plot(title='Training Loss', logy=True)
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")

ax = pd.DataFrame(
    {
        'Adversarial': [metric[1] for metric in a_metrics_complete],
        'Discriminator': [metric[1] for metric in d_metrics_complete],
    }
).plot(title='Training Accuracy')
ax.set_xlabel("Epochs")
ax.set_ylabel("Accuracy")
