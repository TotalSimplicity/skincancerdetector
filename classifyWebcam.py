import cv2
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
import numpy as np
from PIL import Image

# Load the model
model = load_model("model85.h5")

class_indices = {'benign':0, 'malignant':1}

# Open the webcam
cap = cv2.VideoCapture(2)

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    
    # Convert the frame to a PIL image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Resize the image
    img = img.resize((300,225))

    # Convert the image to a numpy array
    img_array = np.array(img)

    # Add an extra dimension to the image (since the model expects a batch of images)
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize the image
    img_array = preprocess_input(img_array)

    # Use the model to predict on the image
    predictions = model.predict(img_array)

    # Get the class index with the highest probability
    class_idx = np.argmax(predictions[0])

    # Get the class label from class_indices or class_names
    class_label = list(class_indices.keys())[list(class_indices.values()).index(class_idx)]

    # Display the class label on the webcam frame
    cv2.putText(frame, class_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
