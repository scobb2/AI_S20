import tensorflow
from tensorflow.keras import models, layers
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sys import exit

(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

# Types
print(type(mnist), type(train_data), type(train_data[0]))

image_shape = train_data[0].shape

# Shapes
#print(train_data.shape, train_labels.shape, image_shape)

#Content
#print("First image", train_data[0])
#print("First label", train_labels[0])
#plt.imshow(train_data[0], cmap=plt.cm.gray_r)
#plt.show()

#exit()

num_pixels = image_shape[0]*image_shape[1]

nn = models.Sequential()
nn.add(layers.Dense(512, activation='relu', input_shape=(num_pixels,)))
nn.add(layers.Dense(32, activation='relu'))  # 512 input shape implied
nn.add(layers.Dense(10, activation='softmax')) 

# Process it all, configure parameters, and get ready to train
nn.compile(
    optimizer="rmsprop",             # Improved backprop algorithm
    loss='categorical_crossentropy', # "Misprediction" measure
    metrics=['accuracy']             # Report CCE value as we train
)

nn.summary()

# Reorganize training and test data as sets of flat vectors
train_data = train_data.reshape((train_data.shape[0], num_pixels))
test_data = test_data.reshape((test_data.shape[0], num_pixels))

# Revise pixel data to 0.0 to 1.0, 32-bit float (this isn't quantum science)
train_data = train_data.astype('float32') / 255  # ndarray/scalar op
test_data = test_data.astype('float32') / 255

# Turn 1-value labels to 10-value vectors
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Post-transform
# print(train_data[0])
#print(train_labels[0])
# exit()

# for epoch in range(5):
#   hst = nn.fit(train_data, train_labels, epochs=1, batch_size = 128)
#   loss, acc = nn.evaluate(test_data, test_labels)
#   print("Epoch %d has train accuracy %f and test accuracy %f"
#    % (epoch, hst.history['acc'][0], acc))
    
hst = nn.fit(train_data, train_labels, epochs = 5, batch_size = 128,
 validation_data = (test_data, test_labels))

hst = hst.history
x_axis = range(len(hst['accuracy']))

plt.plot(x_axis, hst['accuracy'], 'bo')
plt.plot(x_axis, hst['val_accuracy'], 'ro')
plt.show()




