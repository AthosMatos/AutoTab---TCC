import tensorflow as tf
import numpy as np
from keras.layers import (
    Conv2D,
    Dense,
    MaxPooling2D,
    Activation,
    Flatten,
    Dropout,
    Input,
    BatchNormalization,
)
from keras.losses import (
    BinaryCrossentropy,
    BinaryFocalCrossentropy,
    CategoricalCrossentropy,
)

train_ds, unique_labels = (
    np.load("notes_np_cqt_44.1k/train_ds-6out.npz"),
    np.load("all_labels.npy"),
)

train_x = train_ds["x"]
# train_y = train_ds["y"]

train_y = []
# fill train_y with all the values from train_ds["y"]
for i in range(len(unique_labels)):
    train_y.append(train_ds["y"][:, i])

input_shape = (train_x.shape[1], train_x.shape[2], 1)
num_labels = len(unique_labels)

print("")

print(f"Training with {len(train_x)} files")
print("")


def convLayer(x, filters, max_pool=(2, 2), pool_strides=(2, 2)):
    x = Conv2D(filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=max_pool, strides=pool_strides)(x)
    x = Dropout(0.25)(x)

    return x


inputs = Input(shape=input_shape)
x = convLayer(inputs, 32)
for conv in [64, 128, 256]:
    x = convLayer(x, conv)

x = Flatten()(x)

lstm_input = Input(shape=(None, train_x.shape[2], 1))
x_lstm = convLayer(lstm_input, 32)
for conv in [64, 128, 256]:
    x_lstm = convLayer(x_lstm, conv)

x_lstm = Flatten()(x_lstm)


x = Dense(256)(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)

x = Dropout(0.5)(x)
""" 
x = Dense(64)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x) """
# outs = Dense(num_labels, activation="sigmoid")(x)

outs = []

for _ in range(num_labels):
    outs.append(Dense(1, activation="sigmoid")(x))

model = tf.keras.Model(inputs, outs)

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=BinaryCrossentropy(),  # BinaryFocalCrossentropy(),
    metrics=["accuracy"],
)
