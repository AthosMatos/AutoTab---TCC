import tensorflow as tf
from keras import Input
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dropout,
    Dense,
    Flatten,
    Resizing,
    BatchNormalization,
    Normalization,
)

from keras.models import Sequential
import numpy as np

train_ds, val_ds, unique_labels = (
    np.load("np_files/train_ds.npz"),
    np.load("np_files/val_ds.npz"),
    np.load("np_files/unique_labels.npy"),
)
# input_shape = (train_ds["x"].shape[1], train_ds["x"].shape[2], 1)
input_shape = (train_ds["x"].shape[1], train_ds["x"].shape[2], 1)
num_labels = len(unique_labels)

""" norm_layer = Normalization()
# Fit the state of the layer to the spectrograms
# with `Normalization.adapt`.
# train_ds["x"] is a numpy array
norm_layer.adapt(data=train_ds["x"]) """

""" 
BEST
model = Sequential(
    [
        Input(shape=input_shape),
        Resizing(128, 128),
        # norm_layer,  # or dilation, in testing yet
        # Conv2D(16, 3, activation="tanh", padding="same", strides=2),
        # BatchNormalization(),
        # MaxPooling2D(),
        Conv2D(32, 3, activation="tanh", padding="same", strides=2),
        MaxPooling2D(),
        Conv2D(64, 3, activation="tanh", padding="same", strides=2),
        MaxPooling2D(),
        # MaxPooling2D(),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation="tanh"),
        Dropout(0.5),
        Dense(num_labels, activation="softmax"),
    ]
)

"""

model = Sequential(
    [
        Input(shape=input_shape),
        Resizing(128, 128),
        Conv2D(32, 3, activation="tanh", padding="same", strides=2),
        MaxPooling2D(),
        Conv2D(64, 3, activation="tanh", padding="same", strides=2),
        MaxPooling2D(),
        # MaxPooling2D(),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation="tanh"),
        Dropout(0.5),
        Dense(num_labels, activation="softmax"),
    ]
)


model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=["accuracy"],
)


# Train the model using the x_train and y_train arrays
model.fit(
    train_ds["x"],
    train_ds["y"],
    validation_data=(val_ds["x"], val_ds["y"]),
    batch_size=128,
    epochs=64,
    # callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
)


# save the model
model.save("model-[64,64]-tanh-[-1,1].keras")
