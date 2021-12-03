from tensorflow.keras import models, layers
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sys import exit

(train_data, train_labels), (test_data, test_labels) = mnist.load_data()
 
# Reorganize training and test data as sets of flat vectors
train_data = train_data.reshape((60000, 28, 28, 1))
test_data = test_data.reshape((10000, 28, 28, 1))

# Revise pixel data to 0.0 to 1.0, 32-bit float (this isn't quantum science)
train_data = train_data.astype('float32') / 255  # ndarray/scalar op
test_data = test_data.astype('float32') / 255

# Turn 1-value labels to 10-value vectors
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

nn = models.load_model('MNIST.model')

prd = nn.predict(train_data[:5])

for out_vec in prd:
    for val in out_vec:
        print("{:.2f}".format(val), end=" ")
    print()

analysisNN = models.Model(inputs=nn.input, outputs=[layer.output for layer in nn.layers])

for idx in range(5):
    plt.imshow(train_data[idx, :, :, 0], cmap=plt.cm.gray_r)
    plt.show()
   
    images_per_row = 8
    for layer in analysisNN.predict(train_data[idx:idx+1])[:5]:
        print(layer.shape)
        width = layer.shape[1]
        height = layer.shape[2]
        num_chls = layer.shape[3]
        num_rows = num_chls // images_per_row
        display = np.zeros((height * num_rows, width * images_per_row))
      
        for row in range(num_rows):
            for col in range(images_per_row):
                image = layer[0, :, :, row*images_per_row + col]
                mean = image.mean()
                std_dev = image.std()
                print("Mean: {0:.3}, StdDev: {1:.3}".format(mean, std_dev))
                image -= mean
                image /= std_dev
                image *= 64
                image += 120
                image = np.clip(image, 0, 255).astype('uint8')
                display[row*height:(row+1)*height, col*width:(col+1)*width] = image
                 
        plt.imshow(display, cmap=plt.cm.gray)
        #plt.grid(False)
        plt.show()
        