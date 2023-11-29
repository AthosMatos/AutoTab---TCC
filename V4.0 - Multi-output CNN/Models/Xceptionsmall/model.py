import tensorflow as tf
import numpy as np

train_ds, unique_labels = (
    np.load("np_ds-transposed-new/train_ds-6out.npz"),
    np.load("np_ds-transposed-new/unique_labels-6out.npy"),
)


# input_shape = (train_ds["x"].shape[1], train_ds["x"].shape[2], 1)
input_shape = (train_ds["x"].shape[1], train_ds["x"].shape[2], 1)
num_labels = len(unique_labels)

""" norm_layer = tf.keras.tf.keras.layers.Normalization()
norm_layer.adapt(data=train_ds["x"]) """


inputs = tf.keras.Input(shape=input_shape)

# Entry block
x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation("relu")(x)

previous_block_activation = x  # Set aside residual

for size in [64, 128, 256]:
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.SeparableConv2D(size, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.SeparableConv2D(size, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)

    # Project residual
    residual = tf.keras.layers.Conv2D(size, 1, strides=2, padding="same")(
        previous_block_activation
    )
    x = tf.keras.layers.add([x, residual])  # Add back residual
    previous_block_activation = x  # Set aside next residual

x = tf.keras.layers.SeparableConv2D(512, 3, padding="same")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation("relu")(x)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

activation = "sigmoid"
units = 1

x = tf.keras.layers.Dropout(0.5)(x)


outs = tf.keras.layers.Dense(len(unique_labels), activation=activation)(x)

""" 
outs = []
for labls in unique_labels:
    outs.append(tf.keras.layers.Dense(units, activation=activation)(x)) """


model = tf.keras.Model(inputs, outs)
model.summary()
tf.keras.utils.plot_model(
    model, show_shapes=True, show_layer_names=True, show_dtype=True, expand_nested=True
)

exit()
model.compile(
    # optimizer=tf.keras.optimizers.RMSprop(),
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=["accuracy"],
)

# Train the model using the x_train and y_train arrays
# train_ds["y"] = (numofdata,6,50) so split it to 6 arrays

""" train_y = []
# fill train_y with all the values from train_ds["y"]
for i in range(len(unique_labels)):
    train_y.append(train_ds["y"][:, i]) """


model.fit(
    train_ds["x"],
    train_ds["y"],
    batch_size=128,
    epochs=4,
    # callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
)


# save the model
model.save("model-out-6-Adam-new.h5")
