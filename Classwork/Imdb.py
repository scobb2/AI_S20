import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers, optimizers
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import imdb
import numpy as np

class SequenceInterpreter:
   def __init__(self, imdb):
      word_index = imdb.get_word_index()                                  
      print(word_index["brilliant"], word_index["film"]);
      self.reverse_word_index = dict(
       [(num + 3, word) for (word, num) in word_index.items()])            

   def interpret(self, seq):
      print(' '.join(
       [self.reverse_word_index.get(i, '?') for i in seq]))

seq_int = SequenceInterpreter(imdb)

(train_data, train_labels), (test_data, test_labels) \
 = imdb.load_data(num_words=10000)

print(train_data[0])
seq_int.interpret(train_data[0])
print(train_labels[0])

# exit()

# Generate and return a one-hot version 
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))    
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.    # second index hits *many* columns                        
    return results

# Vectorize data
train_data = vectorize_sequences(train_data)                  
train_labels = np.asarray(train_labels).astype('float32')

test_data = vectorize_sequences(test_data)  
test_labels = np.asarray(test_labels).astype('float32')

# Reserve 10000 samples of train data for validation
val_data, train_data = test_data[:10000], train_data[10000:]
val_labels, train_labels = test_labels[:10000], train_labels[10000:]

nn = models.Sequential()
nn.add(layers.Dense(16, activation='relu',  input_shape=(10000,)
#  , kernel_regularizer=regularizers.l2(.001) # Regularize weights
 )
)
# nn.add(layers.Dropout(.5))
nn.add(layers.Dense(16, activation='relu'))
# nn.add(layers.Dropout(.5))
nn.add(layers.Dense(1, activation='sigmoid'))

nn.compile(
    optimizer=optimizers.RMSprop(lr=.001),    # Improved backprop algorithm
    loss='binary_crossentropy',               # "Misprediction" measure
    metrics=['accuracy']                      # Report BCE value as we train
)

hst = nn.fit(train_data, train_labels, epochs = 8, batch_size = 512,
 validation_data = (val_data, val_labels))

hst = hst.history
x_axis = range(0, len(hst['accuracy']))

plt.plot(x_axis, hst['accuracy'], 'bo')
plt.plot(x_axis, hst['val_accuracy'], 'ro')
plt.savefig("Baseline16DO.png")
