import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, Activation
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import numpy as np
from keras.preprocessing import image
from keras import regularizers
import matplotlib.pyplot as plt

model = Sequential()

def createCNN():
  model.add(Conv2D(128, (3,3), input_shape = (180, 180, 3), padding='same', activation = 'relu'))
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size = (2, 2)))
  model.add(Dropout(0.25))

  model.add(Conv2D(256, (3, 3), padding='same', activation = 'relu'))
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size = (2, 2)))
  model.add(Dropout(0.50))

  model.add(Conv2D(256, (3, 3), padding='same', activation = 'relu'))
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size = (2, 2)))
  model.add(Dropout(0.50))
  
  model.add(Conv2D(128, (3,3), padding='same', activation = 'relu'))
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size = (2, 2)))
  model.add(Dropout(0.25))

  model.add(Flatten())

def createNeuralNetwork():

  model.add(Dense(units = 256, activation = 'relu', kernel_initializer = 'normal'))
  model.add(Dropout(0.25))

  model.add(Dense(units = 512, activation = 'relu', kernel_initializer = 'normal'))
  model.add(Dropout(0.50))
  
  model.add(Dense(units = 384, activation = 'relu', kernel_initializer = 'normal'))
  model.add(Dropout(0.25))
  
  model.add(Dense(units = 256, activation = 'relu', kernel_initializer = 'normal'))
  model.add(Dropout(0.30))

  model.add(Dense(units = 512, activation = 'relu', kernel_initializer = 'normal'))
  model.add(Dropout(0.30))

  #Output layer with two units because currenty we have two class of objects
  model.add(Dense(units = 2, activation = 'sigmoid'))

  # Setting some configurations before training
  opt = keras.optimizers.Adam(learning_rate=0.001)
  model.compile(optimizer = opt, 
                loss = 'binary_crossentropy', 
                metrics = ['accuracy'])

def trainingCNN():
  # Augmentation process
  trainingGenerator = ImageDataGenerator(rescale=1. / 255,
                                        rotation_range=20,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        horizontal_flip=True)

  validationGenerator = ImageDataGenerator(rescale=1. / 255)

  # Generate database - Training base and validation base
  trainingData = trainingGenerator.flow_from_directory('Dataset/training',
                                                        target_size = (180, 180),
                                                        batch_size = 16,
                                                        class_mode = 'categorical',
                                                        shuffle=True)

  validationData = validationGenerator.flow_from_directory('Dataset/validation',
                                                target_size = (180, 180),
                                                batch_size = 16,
                                                class_mode = 'categorical',
                                                shuffle=True)
  
  filepath="./out/{val_accuracy:.4f}.hdf5"
  checkpoint = ModelCheckpoint(filepath, 
                                monitor = 'val_accuracy', 
                                verbose = 1, 
                                save_best_only = True, 
                                mode = 'max')

  callbacks_list = [
    checkpoint
  ]

  model.summary()

  # Training the Convolutional Neural Network
  history = model.fit(trainingData, 
                      epochs = 100,
                      validation_data = validationData, 
                      callbacks = callbacks_list)

  # summarize history for accuracy
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.show()

  # summarize history for loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.show()
  
def main():
  createCNN()
  createNeuralNetwork()
  trainingCNN()
    
if (__name__ == '__main__'):
  main()