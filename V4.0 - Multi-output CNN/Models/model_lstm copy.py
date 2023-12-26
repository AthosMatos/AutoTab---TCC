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
    GlobalAveragePooling2D,
    Resizing,
    TimeDistributed,
    LSTM,
    RepeatVector,
    Bidirectional,
)
from keras.losses import (
    BinaryCrossentropy,
    BinaryFocalCrossentropy,
    CategoricalCrossentropy,
)


train_ds, unique_labels = (
    np.load("chords_np_cqt_44.1k/train_ds-6out.npz"),
    np.load("all_labels.npy"),
)

train_x = train_ds["x"]
train_y = train_ds["y"]

print(train_x.shape)
print(train_y.shape)

""" train_y = []
# fill train_y with all the values from train_ds["y"]
for i in range(len(unique_labels)):
    train_y.append(train_ds["y"][:, i]) """

input_shape = (train_x.shape[1], train_x.shape[2], 1)
num_labels = len(unique_labels)

print("")

print(f"Training with {len(train_x)} files")
print("")


inputs = Input(shape=input_shape)
x = Conv2D(32, 3, padding="same", activation="relu")(inputs)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
x = Dropout(0.25)(x)

x = Conv2D(64, 3, padding="same", activation="relu")(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
x = Dropout(0.25)(x)

x = Conv2D(128, 3, padding="same", activation="relu")(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
x = Dropout(0.25)(x)

latent_dim = 128

encoder_input = TimeDistributed(Flatten())(x)

bidirectional_encoder = Bidirectional(LSTM(units=latent_dim, return_state=True))

# The output will be a tuple (output_sequence, forward_state, backward_state)
(
    encoder_output,
    forward_state_h,
    forward_state_c,
    backward_state_h,
    backward_state_c,
) = bidirectional_encoder(encoder_input)


# generate a start decoder input with shape (time_steps, latent_dim) -> (num_labels, latent_dim) -> (num_labels, 128)
decoder_input = RepeatVector(num_labels)(encoder_output)

decoder = LSTM(latent_dim, return_sequences=True, dropout=0.25)
decoder_output = decoder(
    decoder_input,
    initial_state=[
        forward_state_h,
        forward_state_c,
    ],
)

decoder2 = LSTM(latent_dim, return_sequences=True, dropout=0.25, go_backwards=True)
decoder_output2 = decoder2(
    decoder_input,
    initial_state=[
        backward_state_h,
        backward_state_c,
    ],
)

decoder_output = tf.keras.layers.concatenate([decoder_output, decoder_output2])

decoder_dense = TimeDistributed(Dense(1, activation="sigmoid"))
decoder_outputs = decoder_dense(decoder_output)

model = tf.keras.Model(inputs, decoder_outputs)


model.summary()

# plot model
from keras.utils import plot_model

plot_model(
    model, to_file="model_lstm_plot.png", show_shapes=True, show_layer_names=True
)


exit()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=BinaryCrossentropy(),  # BinaryFocalCrossentropy(),
    metrics=["accuracy"],
)

model.fit(
    train_x,
    train_y,
    validation_split=0.2,
    batch_size=128,
    epochs=28,
    verbose=1,
    # callbacks=callbacks
)

# save the model
model.save("/content/drive/MyDrive/AutoTAB/Models/model_chords_lstm.h5")
