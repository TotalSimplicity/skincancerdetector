import tensorflow as tf
import numpy as np
import os

# Set the path to the directory containing the images
data_dir = 'C:/Users/leona/Documents/Projects/Full/images'

# Get the list of class names
class_names = os.listdir(data_dir)

# Set the number of classes
num_classes = len(class_names)

print("Loading to ram.")

# Load the images and labels into memory
images = []
labels = []

print("Loaded.")

for label, class_name in enumerate(class_names):
    for image_name in os.listdir(f'{data_dir}/{class_name}'):
        # Load the image and resize it to (224, 224)
        image = tf.keras.preprocessing.image.load_img(f'{data_dir}/{class_name}/{image_name}', target_size=(224, 224))
        # Convert the image to a numpy array
        image = tf.keras.preprocessing.image.img_to_array(image)
        # Add the image to the list of images
        images.append(image)
        # Add the label to the list of labels
        labels.append(label)
        print("Image processed")
print("Done processing")
# Convert the lists to NumPy arrays
images = np.array(images)
labels = np.array(labels)
print("splitting set")
# Split the data into a training set and a validation set
split = int(0.8 * len(images))
x_train = images[:split]
y_train = labels[:split]
x_val = images[split:]
y_val = labels[split:]
print("Split, creating model")
# Create the model
model = tf.keras.Sequential()
model.add(tf.keras.applications.VGG16(include_top=False, input_shape=(224, 224, 3)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
print("Compiling model")
# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("Training")
# Train the model
model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))
print("Testing Accuracy")
# Test the model on the validation set
loss, accuracy = model.evaluate(x_val, y_val)

print(f'Loss: {loss:.4f}')
print(f'Accuracy: {accuracy:.4f}')

