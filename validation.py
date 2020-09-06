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
from sklearn.metrics import confusion_matrix

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

model.summary()
loss, acc = model.evaluate(testData, verbose=0)
print('\n' + '=============================\n')
print('Accuracy = ' + str(acc) + '\n')
print('Loss = ' + str(loss) + '\n')
print('=============================')

predict = model.predict(testData)
y_predict = []

for x in range(0, len(predict)):
  prediction_class = 0 if predict[x][0] > predict[x][1] else 1
  y_predict.append(prediction_class)

print(y_predict)

conf_matrix = confusion_matrix(testData.classes, y_predict)
print(conf_matrix)
