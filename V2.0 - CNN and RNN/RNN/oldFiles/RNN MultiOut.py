import numpy as np
from keras.models import Sequential, Model
from keras.layers import (
    LSTM,
    TimeDistributed,
    Dense,
    Input,
    Flatten,
    Reshape,
    Conv2D,
    MaxPooling2D,
    Dropout,
)
from DatasetLoader import xTrain, notes, times
import tensorflow as tf
from labels import LABELS
from audioUtils import loadAndPrepare
from paths import DSPATH

""" print(notes.shape)
print(times.shape) """
# Create a neural network with LSTM and TimeDistributed layers

input_layer = Input(shape=(None, xTrain.shape[2], xTrain.shape[3], xTrain.shape[4]))
x = TimeDistributed(Conv2D(32, (3, 3), activation="relu"))(input_layer)
x = TimeDistributed(MaxPooling2D((2, 2)))(x)
x = Dropout(0.25)(x)
x = TimeDistributed(Conv2D(64, (3, 3), activation="relu"))(x)
x = TimeDistributed(MaxPooling2D((2, 2)))(x)
x = Dropout(0.25)(x)
x = TimeDistributed(Flatten())(x)
x = TimeDistributed(Dense(128, activation="relu"))(x)
x = TimeDistributed(Dense(64, activation="relu"))(x)
x = Dropout(0.5)(x)
x = LSTM(128, return_sequences=True)(x)
x = TimeDistributed(Dense(64, activation="relu"))(x)

out1 = TimeDistributed(Dense(notes.shape[-1] * notes.shape[-2], activation="softmax"))(
    x
)
out1 = TimeDistributed(Reshape((notes.shape[-2], notes.shape[-1])))(out1)
out2 = TimeDistributed(Dense(10))(x)

model = Model(input_layer, [out1, out2])

# Compile the model
model.compile(
    optimizer="adam",
    loss=[tf.keras.losses.CategoricalCrossentropy(), "mse"],
    metrics=["accuracy"],
)

model.summary()
n_batch = 1
n_epoch = 250
# Train the model
model.fit(xTrain, [notes, times], epochs=n_epoch, batch_size=n_batch)


test, sr = loadAndPrepare(DSPATH + "TESTMULTINOTES.wav")

TEST = test.reshape(1, 1, test.shape[0], test.shape[1], 1)


# Predict the output for the input data
result = model.predict(TEST, verbose=0)

# Print the result
print(result[0].shape)
print(result[1].shape)
output1_array = 0
output2_array = 1
batchTest = 0
for i, res in enumerate(result[output1_array][batchTest]):
    print(f"|| Audio event {i+1} ||\n")
    for j, ev in enumerate(res):
        timestep = np.round(result[output2_array][batchTest][i][j])
        if timestep >= 0:
            timestep = np.abs(timestep)
        print(f"Time step {timestep} {LABELS[int(np.argmax(ev))]}")


""" print()
for res in result[output2_array][batchTest]:
    print(np.round(res)) """
