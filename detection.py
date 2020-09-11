import cv2
import argparse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from keras.preprocessing import image

ap = argparse.ArgumentParser()
  
ap.add_argument('-m', '--model', required=True,
  help='saved model relative path')

ap.add_argument('-i', '--image', required=True,
  help='Image to detect')

args = vars(ap.parse_args())

# Load the Model
model = load_model(args['model']) 

# Create the haar cascade
frontal_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface.xml')
# profile_face_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')

# Read the image
img = cv2.imread(args['image'])
img = cv2.resize(img, (300, 300))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = frontal_face_cascade.detectMultiScale(
  gray,
  scaleFactor=1.3,
  minNeighbors=5,
  minSize=(5, 5),
  flags=cv2.CASCADE_SCALE_IMAGE 
)

for (x, y, w, h) in faces:
  # resize image to (180,180)
  face_crop = img[y:y+h, x:x+w]
  face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

  # cv2.imshow('test', testImage)
  face_crop = cv2.resize(face_crop, (180, 180))
  face_crop = img_to_array(face_crop)
  face_crop /= 255 
  face_crop = np.expand_dims(face_crop, axis = 0)
  
  (with_mask, without_mask) = model.predict(face_crop)[0]
  label = 'With Mask' if with_mask > without_mask else 'Without Mask'
  color = (0, 255, 0) if label == 'With Mask' else (0, 0, 255)

  # display the label and bounding box rectangle on the output
	# frame
  cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
  cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)

if (len(faces) == 0):
  img_to_predict = image.load_img(args['image'], target_size = (180, 180))
  img_to_predict = image.img_to_array(img_to_predict)
  img_to_predict /= 255 

  img_to_predict = np.expand_dims(img_to_predict, axis = 0)
  (with_mask, without_mask) = model.predict(img_to_predict)[0]
  label = 'With Mask' if with_mask > without_mask else 'Without Mask'
  color = (0, 255, 0) if label == 'With Mask' else (0, 0, 255)
  cv2.putText(img, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
  cv2.rectangle(img, (0, 0), (img.shape[0], img.shape[1]), color, 2)

cv2.imshow("Faces found", img)
cv2.imwrite('detected.png', img)
cv2.waitKey(0)