import pathlib

import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers
from keras.models import Sequential
from tensorflow import keras

def main():
    create_and_train_model()
def create_and_train_model():
    # Load the data from the subfolders at /images/
    data_dir = pathlib.Path("images")
    # Target size - all images will be resized to this size
    target_size = (180, 180)
    input_shape = (target_size[0], target_size[1], 3)

    # Create a dataset
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=target_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=target_size)

    class_names = train_ds.class_names
    num_classes = len(class_names)

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal", input_shape=input_shape),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )

    model = Sequential([
        data_augmentation,
        layers.Rescaling(1. / 255, input_shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.summary()

    model.compile(optimizer='nadam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.build()

    # Train
    epochs = 1
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(epochs)

    model.save('model')

# Code required for the program to run.
if __name__ == "__main__":
    main()
