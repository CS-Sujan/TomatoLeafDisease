import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = "model/tomato_disease_model.keras"

IMAGE_SIZE = (128, 128)

CLASS_NAMES = [
    "Early_Blight",
    "Healthy",
    "Late_Blight",
    "Leaf_Spot"
]

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Ask for image
image_path = input("Enter the FULL path of the tomato leaf image: ").strip('"')

# Load image
image = Image.open(image_path).convert("RGB")

# Resize exactly like training
image = image.resize(IMAGE_SIZE)

# Convert to numpy
image_array = np.array(image, dtype=np.float32)

# IMPORTANT:
# Your model already contains Rescaling(1./255)
# Therefore DO NOT divide by 255 here.

# Add batch dimension
image_array = np.expand_dims(image_array, axis=0)

# Predict
predictions = model.predict(image_array, verbose=0)[0]

predicted_index = np.argmax(predictions)

print("\n==============================")
print(" TOMATO DISEASE PREDICTION")
print("==============================")

print("Prediction :", CLASS_NAMES[predicted_index])
print(f"Confidence : {predictions[predicted_index] * 100:.2f}%")

print("\nAll class probabilities:")

for i, class_name in enumerate(CLASS_NAMES):
    print(f"{class_name:15} : {predictions[i] * 100:.2f}%")

print("==============================")