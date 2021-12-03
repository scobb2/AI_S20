import tensorflow
from keras.models import load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras import models, layers
from tensorflow.keras.utils import to_categorical
from sys import exit


batchSize = 32

# Load data from mnist
(train_data, train_labels), (test_data, test_labels) \
 = mnist.load_data()

# Load best performing model from HPTest
saved_model = tensorflow.keras.models.load_model('best_model.h5')

# Reshape training and test data to add an additional dimension of 1 channel
train_data = train_data.reshape((60000, 28, 28, 1))
test_data = test_data.reshape((10000, 28, 28, 1))

train_data = train_data[:60000]   # range indexing capability (take elements 0 through 5999)
test_data = test_data[:10000]

# Revise pixel data to 0.0 to 1.0, 32-bit float (this isn't quantum science)
train_data = train_data.astype('float32') / 255  # ndarray/scalar op
test_data = test_data.astype('float32') / 255

# Turn 1-value labels to 10-value vectors
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

train_labels = train_labels[:60000]
test_labels = test_labels[:10000]

# Evaluate the model on the test data
results = saved_model.evaluate(test_data, test_labels, \
 batch_size = batchSize)
print("val loss, val acc:", results)