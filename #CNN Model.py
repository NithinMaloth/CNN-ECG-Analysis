#CNN Model
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Define the paths to your training, validation, and test data
train_data_dir = 'path_to_training_data'
val_data_dir = 'path_to_validation_data'
test_data_dir = 'path_to_test_data'

# Define hyperparameters
batch_size = 32
epochs = 20
input_shape = (64, 64, 1)
num_classes = 2  # Number of classes (normal and myocardial infarction)

# Data augmentation for training data
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Validation and test data generators
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

# Load and prepare the data using generators
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=input_shape[:2],
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_data_dir,
    target_size=input_shape[:2],
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=input_shape[:2],
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical'
)

# Create the CNN model
model = models.Sequential([
    # ... Add convolutional, pooling, and fully connected layers as described earlier
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size
)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print(f"Test accuracy: {test_acc}")

# Save the trained model
model.save('ecg_cnn_model.h5')
