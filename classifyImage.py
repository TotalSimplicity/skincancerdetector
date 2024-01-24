import cv2
import numpy as np
from keras.models import load_model
import sys

droppedFile = sys.argv[1]

# Load the model
model = load_model("model85.h5")

# Load the image you want to predict on
img = cv2.imread(droppedFile)

# Resize the image
img = cv2.resize(img, (300, 225))

# Convert the image to a numpy array
img_array = np.array(img)

# Add an extra dimension to the image (since the model expects a batch of images)
img_array = np.expand_dims(img_array, axis=0)

# Normalize the image
img_array = img_array / 255.

class_indices = {'benign':0, 'malignant':1}

# Use the model to predict on the image
predictions = model.predict(img_array)

# Get the class index with the highest probability
class_idx = np.argmax(predictions[0])

# Get the class label from class_indices or class_names
class_label = list(class_indices.keys())[list(class_indices.values()).index(class_idx)]

print("Predicted class: ", class_label)

# Add the text to the image
cv2.putText(img, class_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

# Display the image
cv2.imshow(class_label, img)

# Wait for a key press
cv2.waitKey(0)

# Close the window
cv2.destroyAllWindows()
