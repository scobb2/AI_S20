from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import mnist

(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

print(train_data.shape)

train_data = train_data.astype('float')/255

print("Original images")
for image in train_data[:5]:
   plt.imshow(image, cmap=plt.cm.gray_r)
   plt.show()

gen = ImageDataGenerator(
 rotation_range=10,
 shear_range=.2,
 zoom_range=.1
)

for bnum, batch in zip(range(5), gen.flow(train_data[:5].reshape(5, 28, 28, 1),
 batch_size=5)):
   print("Batch {}".format(bnum))
   for image in batch:
      plt.imshow(image.reshape(28, 28), cmap=plt.cm.gray_r)
      plt.show()

