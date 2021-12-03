import json
import itertools
import os
import tensorflow
from tensorflow.keras import models, layers
from tensorflow.keras.layers import BatchNormalization
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sys import exit

# Load data from mnist
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()
print(train_data.shape)
print(test_data.shape)

# Path variables
jsonFilePath = "LyrInfo.json"
outFilePath = "HPTest.out" # for the text output of the report from running the tests
mcFilePath = "best_model.h5" # saves the best model based on ModelCheckpoint, based on validation accuracy

outFile = open(outFilePath, "a") # Opens or creates the file to store the report of networks being run


# Variables
numCNNs = 0 # Used for count
maxNumCNNs = 299 # Num ~unique~ CNNs is maxNumCNNs / repsRerTest
repsPerTest = 3
batchSize = 32
maxNumEpochs = 8
# stepsPerEpoch = 1500 # len(train_data)/batch_size
# Sets the training to stop if validation accuracy declines a few epochs in a row (to mprevent some overfitting)
es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 2)
# Saves best performing model based on validation accuracy
mc = ModelCheckpoint('best_model.h5', monitor = 'val_accuracy', mode = 'max', verbose = 1, save_best_only = True)


###########################################
# ADD USING DIFFERENT # OF CONV LAYERS ?? #
###########################################
def setup():
    print("Loading json file...")
    jsonFile = open(jsonFilePath, "r") # Obtain file contents
    jsonParams = json.load(jsonFile)
    numChnls = jsonParams['CNNLyrs']['numChnls']
    drpt = jsonParams['CNNLyrs']['drpt']
    batchNorm = jsonParams['CNNLyrs']['batchNorm']
    jsonFile.close()

    # List comp to delve into each possible parameter
    nnPermutations = [LyrParam(chan1, chan2, chan3, drpt1, drpt2, drpt3, CNNbatchNorm)
        for chan1 in numChnls
        for chan2 in numChnls
        for chan3 in numChnls
        for drpt1 in drpt
        for drpt2 in drpt
        for drpt3 in drpt
        for CNNbatchNorm in batchNorm
        if numCNNs <= maxNumCNNs # Used to stop making CNNs once there are x #s to test
    ]
    outFile.close()
    return nnPermutations


 # Describes, for each layer, the number of channels, a dropout rate (or none), and a regularization choice. 
 # A list of LyrParam objects can describe configuration for the entire CNN, one LyrParam per convo layer.
class LyrParam :
    def __init__(self, chan1, chan2, chan3, drpt1, drpt2, drpt3, CNNbatchNorm):
        global numCNNs # to use numCNNs variable
        self.chan1 = chan1
        self.chan2 = chan2
        self.chan3 = chan3
        self.drpt1 = drpt1
        self.drpt2 = drpt2
        self.drpt3 = drpt3
        self.CNNbatchNorm = CNNbatchNorm
        self.returnData = []

        self.nn = [None] * repsPerTest
        for x in range(repsPerTest):
            self.nn[x] = models.Sequential() # Create nn model
            self.nn[x].add(layers.Conv2D(self.chan1, (3, 3), activation = 'relu', input_shape = (28, 28, 1)))  # Add Conv layer
            if self.CNNbatchNorm == True: self.nn[x].add(BatchNormalization())  # Check for batch norm
            self.nn[x].add(layers.MaxPooling2D((2, 2))) # Add pooling layer
            self.nn[x].add(layers.Dropout(self.drpt1)) # Add dropout
            # self.returnData.append("Channels: " + str(self.chan1) + " Dropout: " 
            #     + str(self.drpt1) + " Batch Normalization: " + str(self.CNNbatchNorm))
            if x == 0: outFile.write("Channels: " + str(self.chan1) + " Dropout: " 
                + str(self.drpt1) + " Batch Normalization: " + str(self.CNNbatchNorm) + "\n")

            self.nn[x].add(layers.Conv2D(self.chan2, (3, 3), activation = 'relu'))  # Add Conv layer
            if self.CNNbatchNorm == True: self.nn[x].add(BatchNormalization())  # Check for batch norm
            self.nn[x].add(layers.MaxPooling2D((2, 2))) # Add pooling layer
            self.nn[x].add(layers.Dropout(self.drpt2)) # Add dropout
            # self.returnData.append("Channels: " + str(self.chan2) + " Dropout: " 
            #     + str(self.drpt2) + " Batch Normalization: " + str(self.CNNbatchNorm))
            if x == 0: outFile.write("Channels: " + str(self.chan2) + " Dropout: " 
                + str(self.drpt2) + " Batch Normalization: " + str(self.CNNbatchNorm) + "\n")

            self.nn[x].add(layers.Conv2D(self.chan3, (3, 3), activation = 'relu'))  # Add Conv layer
            if self.CNNbatchNorm == True: self.nn[x].add(BatchNormalization())  # Check for batch norm
            self.nn[x].add(layers.MaxPooling2D((2, 2))) # Add pooling layer
            self.nn[x].add(layers.Dropout(self.drpt3)) # Add dropout
            # self.returnData.append("Channels: " + str(self.chan1) + " Dropout: " 
            #     + str(self.drpt3) + " Batch Normalization: " + str(self.CNNbatchNorm))
            if x == 0: outFile.write("Channels: " + str(self.chan1) + " Dropout: " 
                + str(self.drpt3) + " Batch Normalization: " + str(self.CNNbatchNorm) + "\n")

            # Always the last layers
            self.nn[x].add(layers.Flatten())  # Output shape (:, 576)
            self.nn[x].add(layers.Dense(10, activation='softmax')) # Output #s 0-9

            # Display the created CNN (every 3 to avoid duplicate printing)
            # if x % 3 == 0: self.nn[x].summary()
            # if x == 0: print("\n\n" + str(self.returnData))

            self.nn[x].compile(
                optimizer="RMSprop",             # Improved backprop algorithm
                loss='categorical_crossentropy', # "Misprediction" measure
                metrics=['accuracy']             # Report CCE value as we train
            )
            # Training on default images:
            # hst = self.nn[x].fit(train_data, train_labels, epochs = maxNumEpochs, batch_size = batchSize, 
            #     validation_data = (test_data, test_labels), callbacks = [es, mc], verbose = 0).history

            # Training on augmented data:
            hst = self.nn[x].fit(x = train_generator, validation_data = validation_generator, epochs = maxNumEpochs, 
                callbacks = [es, mc], verbose = 1).history

            scores = self.nn[x].evaluate(test_data, test_labels, verbose = 10)
            print ( scores )
            #hst = self.nn[x].evaluate(test_data, test_labels, batch_size = batchSize, verbose = 1, callbacks = [es, mc]).history # pred_after_augmented

            print ("Max val accuracy: " + str(max(hst['val_accuracy'])) + " with loss: " + str(min(hst['val_loss'])))
            outFile.write("Max val accuracy: " + str(max(hst['val_accuracy'])) + " with loss: " + str(min(hst['val_loss'])) + "\n")

            numCNNs += 1
        ###############################################################
        # Average the outcomes of 3 tests for each different CNN ???? #
        ###############################################################
        


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

train_labels = train_labels[:60000]
test_labels = test_labels[:10000]


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
print (datagen.flow(train_data, train_labels, batch_size = batchSize))
train_generator = datagen.flow(train_data, train_labels, batch_size = batchSize)
validation_generator = datagen.flow(train_data, train_labels, batch_size = batchSize)

# train_data_augmented = datagen.flow(train_data[:60000].reshape(60000, 28, 28, 1), batch_size=batchSize)
# test_generator = datagen.flow(test_data[:1000].reshape(1000, 28, 28, 1), batch_size=batchSize)



setup()

# exit()





