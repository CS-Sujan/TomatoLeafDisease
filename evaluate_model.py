import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)

# ==============================
# SETTINGS
# ==============================
# ==============================
# SETTINGS
# ==============================

DATASET_PATH = "balanced_dataset"
MODEL_PATH = "model/tomato_disease_model.keras"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

# ==============================
# LOAD VALIDATION DATASET
# ==============================

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

class_names = validation_dataset.class_names

print("\nClasses:")
print(class_names)

# ==============================
# LOAD MODEL
# ==============================

model = tf.keras.models.load_model(MODEL_PATH)

# ==============================
# PREDICTIONS
# ==============================

true_labels = []
predicted_labels = []

for images, labels in validation_dataset:

    predictions = model.predict(images, verbose=0)

    predicted = np.argmax(predictions, axis=1)

    true_labels.extend(labels.numpy())
    predicted_labels.extend(predicted)

true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)

# ==============================
# ACCURACY
# ==============================

accuracy = np.mean(true_labels == predicted_labels)

print("\n==============================")
print(" MODEL ACCURACY")
print("==============================")

print(f"Accuracy: {accuracy * 100:.2f}%")

# ==============================
# CLASSIFICATION REPORT
# ==============================

print("\n==============================")
print(" CLASSIFICATION REPORT")
print("==============================")

report = classification_report(
    true_labels,
    predicted_labels,
    labels=np.arange(len(class_names)),
    target_names=class_names,
    zero_division=0
)

print(report)

# ==============================
# CONFUSION MATRIX
# ==============================

cm = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=np.arange(len(class_names))
)

print("\n==============================")
print(" CONFUSION MATRIX")
print("==============================")

print(cm)

# ==============================
# DISPLAY CONFUSION MATRIX
# ==============================

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(
    xticks_rotation=45
)

plt.title("Tomato Leaf Disease - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300
)

plt.show()

# ==============================
# COMPLETED
# ==============================

print("\n==============================")
print(" EVALUATION COMPLETED")
print("==============================")

print("Confusion matrix saved as:")
print("confusion_matrix.png")