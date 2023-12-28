from keras.models import Model
from keras.optimizers import Adam
from keras.metrics import CategoricalAccuracy, MeanAbsoluteError

from keras.losses import categorical_crossentropy, mean_squared_error
from keras.layers import (
    LSTM,
    TimeDistributed,
    Input,
    Dense,
    RepeatVector,
    Conv2D,
    MaxPooling2D,
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dropout,
    Bidirectional,
    Activation,
    Reshape,
    BatchNormalization,
)
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint, EarlyStopping


# define mode
inputs = Input(shape=(X.shape[1], X.shape[2], X.shape[3]))

x = Conv2D(32, 3, padding="same")(inputs)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
x = Dropout(0.25)(x)

x = Conv2D(64, 3, padding="same")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
x = Dropout(0.25)(x)

x = Conv2D(128, 3, padding="same")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
x = Dropout(0.25)(x)

convoluted_vector = TimeDistributed(Flatten())(x)

encoder_output = Bidirectional(LSTM(256, return_sequences=True, dropout=0.2))(
    convoluted_vector
)
encoder_output = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(
    encoder_output
)


out_notes = TimeDistributed(Dense(128))(encoder_output)
out_notes = TimeDistributed(BatchNormalization())(out_notes)
out_notes = TimeDistributed(Activation("relu"))(out_notes)
out_notes = TimeDistributed(Dropout(0.5))(out_notes)

out_notes = TimeDistributed(
    Dense(notes_outs_len, activation="softmax"), name="notes_output"
)(out_notes)

out_times = TimeDistributed(Dense(128))(encoder_output)
out_times = TimeDistributed(BatchNormalization())(out_times)
out_times = TimeDistributed(Activation("relu"))(out_times)
out_times = TimeDistributed(Dropout(0.5))(out_times)

out_times = TimeDistributed(Dense(2), name="times_output")(out_times)


model = Model(inputs=inputs, outputs=[out_notes, out_times])
metrics = {
    "notes_output": "categorical_accuracy",  # Adjust based on your task
    "times_output": "mean_squared_error",  # Mean Squared Error
}
losses = {
    "notes_output": "categorical_crossentropy",  # Adjust based on your task
    "times_output": "mean_squared_error",  # Mean Squared Error
}
model.compile(
    optimizer="adam",  #'rmsprop'
    loss=losses,  # 'mse
    metrics=metrics,
)
# fit model

model.summary()

# plot model
from keras.utils import plot_model

plot_model(model, to_file="EncoderDecoder.png", show_shapes=True, show_layer_names=True)
