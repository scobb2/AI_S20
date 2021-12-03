import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers, preprocessing
from tensorflow.keras.preprocessing import sequence
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import imdb
import numpy as np

num_unique_words = 5000
embed_dim = 64
review_len = 200

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=num_unique_words)

# Reserve 10000 samples of train data for validation
val_data, train_data = test_data[:10000], train_data[10000:]
val_labels, train_labels = test_labels[:10000], train_labels[10000:]

# Vectorize data
train_data = sequence.pad_sequences(train_data, maxlen=review_len)             
train_labels = np.asarray(train_labels).astype('float32')

val_data = sequence.pad_sequences(val_data, maxlen=review_len)
val_labels = np.asarray(val_labels).astype('float32')

mdl = models.Sequential()
mdl.add(layers.Embedding(num_unique_words, embed_dim, input_length=review_len))
mdl.add(layers.LSTM(32, dropout=.2))
mdl.add(layers.Dense(1, activation='sigmoid'))

mdl.summary()

mdl.compile(
    optimizer='adam',             # Improved backprop algorithm
    loss='binary_crossentropy',
    metrics=['accuracy']   
)

hst = mdl.fit(train_data, train_labels, epochs = 12, batch_size = 512,
 validation_data = (val_data, val_labels))

hst = hst.history
x_axis = range(0, len(hst['accuracy']))

plt.plot(x_axis, hst['accuracy'], 'bo')
plt.plot(x_axis, hst['val_accuracy'], 'ro')
plt.savefig("BaselineIMDBLSTM.png")
