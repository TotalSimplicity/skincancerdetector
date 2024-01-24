import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import glob

data_dir = "C:/Users/leona/Documents/Projects/Full/images"
img_height = 150
img_width = 150
batch_size = 32

# Splitting data into train and validation sets
image_paths = [path for path in glob.glob(data_dir + '/*')]
train_paths, val_paths = train_test_split(image_paths, test_size=0.2, random_state=42)

# Data augmentation
data_augmentation = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# Generators for training and validation data
train_ds = data_augmentation.flow_from_directory(
    data_dir,
    subset='training',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)


val_ds = data_augmentation.flow_from_directory(
    data_dir,
    subset='validation',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)


# Create model
model = keras.Sequential(
    [
        layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(img_height, img_width, 3)),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(train_ds.num_classes, activation='softmax')
    ]
)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit model
epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    validation_steps = 30
)

# Evaluation of the model on the validation set
val_loss, val_acc = model.evaluate(val_ds)
print("Validation Loss: ",val_loss)
print("Validation Accuracy: ",val_acc)
