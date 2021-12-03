import os
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, models, layers
import sys
from matplotlib import pyplot as plt

def load_model(name): 
    print("Loading ", name)
    model = models.load_model(name)
    print(model.layers[0].summary())
    print(model.summary())

    return model

def release_layers(model, layer_names):
    cnv = model.layers[0]
    for layer in cnv.layers:
        layer.trainable = (layer.name in layer_names)
    print(cnv.summary())
    
    model.compile(loss='binary_crossentropy',
     optimizer=optimizers.RMSprop(lr=2e-5),
     metrics=['acc'])

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
        sys.argv[2],              # Directory containing train data
        target_size=(150, 150), 
        batch_size=20,
        class_mode='binary')       
    
    
    vld_generator = noaug_datagen.flow_from_directory(
        sys.argv[3],              # Validation data directory
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

    return (train_generator, vld_generator)
    
def main():
    model = load_model(sys.argv[1])
    release_layers(model, sys.argv[4].split(','))

    train_generator, vld_generator = make_generators()

    hst = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=30,
        validation_data=vld_generator,
        validation_steps=50).history      
    
    model.save(sys.argv[5])
    
    for acc, loss, val_acc, val_loss in zip(hst['acc'], hst['loss'],
     hst['val_acc'], hst['val_loss']): 
        print("%.5f / %.5f  %.5f / %.5f" % (acc, loss, val_acc, val_loss))

main()
