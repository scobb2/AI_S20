from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import mnist

(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

train_data = train_data[:4]
train_labels = train_labels[:4]

train_data = train_data.reshape((4, 28, 28, 1))

print(train_data.shape)

train_data = train_data.astype('float')/255



# Image Generation
datagen = ImageDataGenerator (
    rotation_range = 5,
    shear_range = .2,
    zoom_range = .1,
    width_shift_range = 2,
    height_shift_range = 2
)

# Create, reshape, and range index new images
datagen.fit(train_data)
train_generator = datagen.flow(train_data, train_labels, \
 batch_size = 2)

# Testing Augmentation
for n, i in enumerate(train_generator):
    batch_data = i[0]
    print(n, batch_data[0].shape)