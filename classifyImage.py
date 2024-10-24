import cv2
import numpy as np
from keras.models import load_model
import tkinter as tk
from tkinterdnd2 import TkinterDnD, DND_FILES
from PIL import Image, ImageTk
import os

# Initialize the TkinterDnD window
root = TkinterDnD.Tk()
root.title("Skin Lesion Classifier")
root.geometry("600x400")

# Load the model
model = load_model("model85.h5", compile=False)

class_indices = {'benign': 0, 'malignant': 1}

# Function to process the dropped file
def process_image(file_path):
    # Check if file is a valid image format
    if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Load the image
        img = cv2.imread(file_path)
        img = cv2.resize(img, (300, 225))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Make a prediction
        predictions = model.predict(img_array)
        class_idx = np.argmax(predictions[0])
        class_label = list(class_indices.keys())[list(class_indices.values()).index(class_idx)]

        # Add text to the image
        cv2.putText(img, class_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Display the image with the prediction in OpenCV window
        cv2.imshow("Predicted class: " + class_label, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Please drop a valid image file (.jpg, .jpeg, .png)")

# Function to handle drag-and-drop
def drop(event):
    file_path = event.data
    file_path = file_path.strip('{}')  # Remove curly braces from file path if present
    if os.path.isfile(file_path):
        process_image(file_path)

# Set up the label in the Tkinter window for dragging and dropping
label = tk.Label(root, text="Drag and drop an image file here", width=50, height=10)
label.pack(pady=50)

# Enable drag-and-drop for the label
label.drop_target_register(DND_FILES)
label.dnd_bind('<<Drop>>', drop)

# Start the Tkinter event loop
root.mainloop()
