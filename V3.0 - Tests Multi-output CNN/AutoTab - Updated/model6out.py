import tensorflow as tf
import numpy as np

train_ds, val_ds, unique_labels = (
    np.load("np_files/train_ds-6out.npz"),
    np.load("np_files/val_ds-6out.npz"),
    np.load("np_files/unique_labels-6out.npy"),
)
# input_shape = (train_ds["x"].shape[1], train_ds["x"].shape[2], 1)
input_shape = (train_ds["x"].shape[1], train_ds["x"].shape[2], 1)
num_labels = len(unique_labels)

""" norm_layer = tf.keras.layers.Normalization()
norm_layer.adapt(data=train_ds["x"]) """


""" 

input_layer = tf.keras.Input(shape=input_shape)
# x = tf.keras.layers.Resizing(128, 128)(input_layer)
# x = norm_layer(input_layer)
x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same", strides=2)(
    input_layer
)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Dropout(0.25)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same", strides=2)(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Dropout(0.25)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same", strides=2)(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Dropout(0.25)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.5)(x)
# 6 outputs
output_layer1 = tf.keras.layers.Dense(num_labels, activation="softmax")(x)
output_layer2 = tf.keras.layers.Dense(num_labels, activation="softmax")(x)
output_layer3 = tf.keras.layers.Dense(num_labels, activation="softmax")(x)
output_layer4 = tf.keras.layers.Dense(num_labels, activation="softmax")(x)
output_layer5 = tf.keras.layers.Dense(num_labels, activation="softmax")(x)
output_layer6 = tf.keras.layers.Dense(num_labels, activation="softmax")(x)

model = tf.keras.Model(
    inputs=input_layer,
    outputs=[
        output_layer1,
        output_layer2,
        output_layer3,
        output_layer4,
        output_layer5,
        output_layer6,
    ],
)

model.summary()
tf.keras.utils.plot_model(model, show_shapes=True)


"""


def convLayer(filters, kernel, strides, input_layer):
    x = tf.keras.layers.Conv2D(
        filters, kernel, activation="relu", padding="same", strides=strides
    )(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)

    return x


def denseLayer(units, input_layer):
    x = tf.keras.layers.Dense(units, activation="relu")(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    return x


input_layer = tf.keras.Input(shape=input_shape)
# x = tf.keras.layers.Resizing(128, 128)(input_layer)
# x = norm_layer(input_layer)
x = convLayer(32, 3, 2, input_layer)
x = convLayer(64, 3, 1, x)
x = convLayer(128, 3, 1, x)
x = tf.keras.layers.Flatten()(x)
x = denseLayer(256, x)
x = denseLayer(128, x)
# 6 outputs
output_layer1 = tf.keras.layers.Dense(num_labels, activation="softmax")(x)
output_layer2 = tf.keras.layers.Dense(num_labels, activation="softmax")(x)
output_layer3 = tf.keras.layers.Dense(num_labels, activation="softmax")(x)
output_layer4 = tf.keras.layers.Dense(num_labels, activation="softmax")(x)
output_layer5 = tf.keras.layers.Dense(num_labels, activation="softmax")(x)
output_layer6 = tf.keras.layers.Dense(num_labels, activation="softmax")(x)

model = tf.keras.Model(
    inputs=input_layer,
    outputs=[
        output_layer1,
        output_layer2,
        output_layer3,
        output_layer4,
        output_layer5,
        output_layer6,
    ],
)

model.summary()
tf.keras.utils.plot_model(model, show_shapes=True)


exit()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=["accuracy"],
)

# Train the model using the x_train and y_train arrays
# train_ds["y"] = (numofdata,6,50) so split it to 6 arrays

train_y_1 = train_ds["y"][:, 0, :]
train_y_2 = train_ds["y"][:, 1, :]
train_y_3 = train_ds["y"][:, 2, :]
train_y_4 = train_ds["y"][:, 3, :]
train_y_5 = train_ds["y"][:, 4, :]
train_y_6 = train_ds["y"][:, 5, :]


val_y_1 = val_ds["y"][:, 0, :]
val_y_2 = val_ds["y"][:, 1, :]
val_y_3 = val_ds["y"][:, 2, :]
val_y_4 = val_ds["y"][:, 3, :]
val_y_5 = val_ds["y"][:, 4, :]
val_y_6 = val_ds["y"][:, 5, :]

model.fit(
    train_ds["x"],
    [train_y_1, train_y_2, train_y_3, train_y_4, train_y_5, train_y_6],
    validation_data=(
        val_ds["x"],
        [val_y_1, val_y_2, val_y_3, val_y_4, val_y_5, val_y_6],
    ),
    batch_size=128,
    epochs=64,
    verbose=2
    # callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
)


# save the model
model.save("model-[128,128]-tanh-[-1,1]-out6.keras")
