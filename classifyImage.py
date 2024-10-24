import cv2
import numpy as np
from keras.models import load_model
import tkinter as tk
from tkinterdnd2 import TkinterDnD, DND_FILES
from PIL import Image, ImageTk
import os


root = TkinterDnD.Tk()
root.title("Skin Lesion Classifier")
root.geometry("600x400")

# Load the model
model = load_model("model85.h5", compile=False)

class_indices = {'benign': 0, 'malignant': 1}


def process_image(file_path):
    # Check if file is an image
    if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        img = cv2.imread(file_path)
        img = cv2.resize(img, (300, 225))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0


        predictions = model.predict(img_array)
        class_idx = np.argmax(predictions[0])
        class_label = list(class_indices.keys())[list(class_indices.values()).index(class_idx)]

        # Add classication label to output image
        cv2.putText(img, class_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


        cv2.imshow("Predicted class: " + class_label, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Please drop a valid image file (.jpg, .jpeg, .png)")


def drop(event):
    file_path = event.data
    file_path = file_path.strip('{}') 
    if os.path.isfile(file_path):
        process_image(file_path)


label = tk.Label(root, text="Drag and drop an image file here", width=50, height=10)
label.pack(pady=50)

label.drop_target_register(DND_FILES)
label.dnd_bind('<<Drop>>', drop)

# Start the Tkinter event loop
root.mainloop()
