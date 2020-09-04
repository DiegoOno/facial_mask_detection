import cv2
import argparse
import keras
from tensorflow.keras.models import load_model
from keras.preprocessing import image
import numpy as np

ap = argparse.ArgumentParser()
  
ap.add_argument('-m', '--model', required=True,
  help='saved model relative path')

ap.add_argument('-i', '--image', required=True,
  help='Image to detect')

args = vars(ap.parse_args())

# Load the Model
model = load_model(args['model'])

testImage = image.load_img(args['image'], target_size = (180, 180))
testImage = image.img_to_array(testImage)
testImage /= 255 

(with_mask, without_mask) = model.predict(np.expand_dims(testImage, axis=0))[0]
label = 'With Mask' if with_mask > without_mask else 'Without Mask'
print(label)