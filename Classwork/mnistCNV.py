from tensorflow.keras import models, layers
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sys import exit

# Convolutional Neural Network (CNN)

(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

print(train_data.shape)

nn = models.Sequential()
nn.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))) # Output shape (.., 26, 26, 32)
nn.add(layers.MaxPooling2D((2, 2)))                  # Output shape (:, 13, 13, 32)
nn.add(layers.Conv2D(64, (3, 3), activation='relu')) # Output shape (:, 11, 11, 64)
nn.add(layers.MaxPooling2D((2, 2)))                  # Output shape (:, 5, 5, 64)
nn.add(layers.Conv2D(64, (3, 3), activation='relu')) # Output shape (:, 3, 3, 64)
nn.add(layers.Flatten())                             # Output shape (:, 576)
nn.add(layers.Dense(64, activation='relu'))          # Output shape (:, 64)
nn.add(layers.Dense(10, activation='softmax')) 

nn.summary()

# Process it all, configure parameters, and get ready to train
nn.compile(
    optimizer="RMSprop",             # Improved backprop algorithm
    loss='categorical_crossentropy', # "Misprediction" measure
    metrics=['accuracy']             # Report CCE value as we train
)

# Reshape training and test data to add an additional dimension of 1 channel
train_data = train_data.reshape((60000, 28, 28, 1))
test_data = test_data.reshape((10000, 28, 28, 1))


train_data = train_data[:60000]   # range indexing capability (take elements 0 through 5999)
test_data = test_data[:1000]

# Revise pixel data to 0.0 to 1.0, 32-bit float (this isn't quantum science)
train_data = train_data.astype('float32') / 255  # ndarray/scalar op
test_data = test_data.astype('float32') / 255

# Turn 1-value labels to 10-value vectors
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

train_labels = train_labels[:60000]
test_labels = test_labels[:1000]

hst = nn.fit(train_data, train_labels, epochs = 8, batch_size = 64,
 validation_data = (test_data, test_labels))

hst = hst.history
x_axis = range(len(hst['accuracy']))

plt.plot(x_axis, hst['accuracy'], 'bo')
plt.plot(x_axis, hst['val_accuracy'], 'ro')
plt.show()

nn.save('MNIST.model')
 