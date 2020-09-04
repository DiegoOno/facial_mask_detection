import numpy as np
import keras
from keras.models import model_from_json
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model
import argparse
import pandas as pd

ap = argparse.ArgumentParser()
  
ap.add_argument('-m', '--model', required=True,
  help='saved model relative path')

args = vars(ap.parse_args())

model = load_model(args['model'])

testGenerator = ImageDataGenerator(rescale=1. / 255)
testData = testGenerator.flow_from_directory('Dataset/test',
                                                 target_size=(180, 180),
                                                 batch_size=1,
                                                 class_mode='categorical',
                                                 shuffle=False)

print(testData.classes)

model.summary()
loss, acc = model.evaluate(testData, verbose=0)
print('\n' + '=============================\n')
print('Accuracy = ' + str(acc) + '\n')
print('Loss = ' + str(loss) + '\n')
print('=============================')