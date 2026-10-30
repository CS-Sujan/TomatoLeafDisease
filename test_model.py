import os
import sys

import numpy as np
import tensorflow as tf
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("model/tomato_disease_model.keras")

# Class names
class_names = [
    "Early_Blight",
    "Healthy",
    "Late_Blight",
    "Leaf_Spot",
]

# Ask for image path
image_path = input(
    "Enter the FULL path of the tomato leaf image: "
).strip().strip('"')

# Check whether file exists
if not os.path.isfile(image_path):
    print("\nERROR: Image file not found.")
    print("You entered:", image_path)
    print("\nExample:")
    print(
        r"C:\Users\ds582\Downloads\TomatoLeafDisease\test_images\leaf.jpg"
    )
    sys.exit()

# Load image
image = Image.open(image_path).convert("RGB")
image = image.resize((128, 128))

# Convert to array
image_array = np.array(image)
image_array = np.expand_dims(image_array, axis=0)

# IMPORTANT:
# The trained model already contains Rescaling(1./255).
# Therefore, do NOT divide by 255 here.

# Predict
prediction = model.predict(image_array, verbose=0)

predicted_index = np.argmax(prediction[0])
predicted_class = class_names[predicted_index]
confidence = prediction[0][predicted_index] * 100

print("\n==============================")
print(" TOMATO DISEASE PREDICTION")
print("==============================")
print("Prediction :", predicted_class)
print("Confidence :", f"{confidence:.2f}%")
print("==============================")