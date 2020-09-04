import cv2
import argparse
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

ap = argparse.ArgumentParser()
  
ap.add_argument('-m', '--model', required=True,
  help='saved model relative path')

ap.add_argument('-i', '--image', required=True,
  help='Image to detect')

ap.add_argument('-c', '--cascade', required=True,
  help='Cascade Names')  

args = vars(ap.parse_args())

# Load the Model
model = load_model(args['model']) 

# Create the haar cascade
face_cascade = cv2.CascadeClassifier(args['cascade'])

# Read the image
image = cv2.imread(args['image'])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(
  gray,
  scaleFactor=1.1,
  minNeighbors=5,
  minSize=(5, 5),
  flags=cv2.CASCADE_SCALE_IMAGE 
)

cnn_input_dim = (180, 180)

print("%d" % len(faces))
# Draw a rectangle around the faces

for (x, y, w, h) in faces:
  # resize image to (180,180)
  face_crop = image[y:y+h, x:x+w]
  face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

  # cv2.imshow('test', testImage)
  face_crop = cv2.resize(face_crop, (180, 180))
  face_crop = img_to_array(face_crop)
  # face_crop /= 255 
  face_crop = np.expand_dims(face_crop, axis = 0)
  
  (with_mask, without_mask) = model.predict(face_crop)[0]
  label = 'With Mask' if with_mask > without_mask else 'Without Mask'
  color = (0, 255, 0) if label == 'With Mask' else (0, 0, 255)

  # include the probability in the label
  # label = "{}: {:.4f}%".format(label, max(with_mask, without_mask) * 100)

  # display the label and bounding box rectangle on the output
	# frame
  cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
  cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)