import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras import optimizers, models, layers
import sys
from matplotlib import pyplot as plt

def make_model(): 
    model = models.Sequential()

    # Load cnv-only part of VGG16
    cnv_layers =  VGG16(weights='imagenet', include_top=False,
     input_shape=(150,150,3))

    # Lock it down so we don't mess it up while training
    cnv_layers.trainable = False
    print(cnv_layers.summary())

    # Start with convolutional portion of VGG16
    model.add(cnv_layers)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy',
     optimizer=optimizers.RMSprop(lr=2e-5),
     metrics=['accuracy'])

    return model

def dump_generator(gen, num_batches = 1):
    for bnum, batch in zip(range(num_batches), gen):
       print("Batch %d" % (bnum), flush=True)
       for image, label in zip(batch[0], batch[1]):
           plt.imshow(image, cmap=plt.cm.gray_r)
           plt.title(label)
           plt.show()
    
def make_generators():
    # Big rotations, shifts, shears, flips.  Lots of varied 
    # (and even funny looking) cats and dogs for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    # Simple scaling and default batchsize.  Don't augment nontraining data
    noaug_datagen = ImageDataGenerator(rescale=1./255)  

    train_generator = train_datagen.flow_from_directory(
        sys.argv[1],              # Directory containing train data
        target_size=(150, 150), 
        batch_size=20,
        class_mode='binary')       
    
    vld_generator = noaug_datagen.flow_from_directory(
        sys.argv[2],              # Validation data directory
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

    # dump_generator(vld_generator)
    # dump_generator(train_generator)

    return (train_generator, vld_generator)
    
def main():
    model = make_model()
    train_generator, vld_generator = make_generators()

    hst = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=20,
        validation_data=vld_generator,
        validation_steps=50).history
    
    model.save(sys.argv[3])
    
    for acc, loss, val_acc, val_loss in zip(hst['accuracy'], hst['loss'],
     hst['val_accuracy'], hst['val_loss']): 
        print("%.5f / %.5f  %.5f / %.5f" % (acc, loss, val_acc, val_loss))

main()
