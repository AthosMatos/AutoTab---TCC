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
    BatchNormalization,
)
import numpy as np
from keras.preprocessing.text import Tokenizer
from utils.notes import genNotes_v2
import matplotlib.pyplot as plt

train_ds = np.load("seq2seqNpy.npz")
GUITAR_NOTES = genNotes_v2("F#1", "A6")

X = train_ds["X"]
y_notes = train_ds["y_notes"]
y_times = train_ds["y_times"]


print(X.shape)
print(y_notes.shape)
print(y_times.shape)
max_seq = y_notes.shape[1]
notes_outs_len = y_notes.shape[2]


note_tokenizer = Tokenizer(lower=False, filters="")
note_tokenizer.fit_on_texts(GUITAR_NOTES)
note_tokenizer = note_tokenizer.word_index

# define mode
inputs = Input(shape=(X.shape[1], X.shape[2], X.shape[3]))

""" x = TimeDistributed(Conv1D(32, 3, activation="relu"))(inputs)
#x = TimeDistributed(BatchNormalization())(x)
#x = TimeDistributed(Activation("relu"))(x)
x = TimeDistributed(MaxPooling1D())(x)

x = TimeDistributed(Conv1D(64, 3, activation="relu"))(x)
#x = TimeDistributed(BatchNormalization())(x)
#x = TimeDistributed(Activation("relu"))(x)
x = TimeDistributed(MaxPooling1D())(x)

x = TimeDistributed(Conv1D(128, 3, activation="relu"))(x)
#x = TimeDistributed(BatchNormalization())(x)
#x = TimeDistributed(Activation("relu"))(x)
x = TimeDistributed(MaxPooling1D())(x) """

""" 
"""

# x = MaxPooling2D(strides=(2, 2))(x)
""" x = Conv2D(32, 3, activation="relu")(inputs)
x = MaxPooling2D()(x)

x = Conv2D(64, 3, activation="relu")(x)
x = MaxPooling2D()(x)

x = Conv2D(128, 3, activation="relu")(x)
x = MaxPooling2D()(x)
 """
# x = Dropout(0.2)(x)
# best one so far without batch normalization
x = TimeDistributed(Conv1D(32, 3, activation="relu"))(inputs)
x = TimeDistributed(MaxPooling1D())(x)
x = TimeDistributed(Conv1D(64, 3, activation="relu"))(x)
x = TimeDistributed(MaxPooling1D())(x)
x = TimeDistributed(Conv1D(128, 3, activation="relu"))(x)
x = TimeDistributed(MaxPooling1D())(x)

x = TimeDistributed(Flatten())(x)

x = Bidirectional(LSTM(128))(x)
x = RepeatVector(max_seq)(x)
x = Bidirectional(LSTM(256, return_sequences=True))(x)
""" x = LSTM(128)(x)
x = RepeatVector(max_seq)(x)
x = LSTM(256, return_sequences=True)(x) """


out_notes = TimeDistributed(Dense(256, activation="relu"))(x)
out_notes = TimeDistributed(Dense(128, activation="relu"))(out_notes)
out_notes = TimeDistributed(Dense(84, activation="relu"))(out_notes)
# out_notes = TimeDistributed(Dropout(0.5))(out_notes)
out_notes = TimeDistributed(
    Dense(notes_outs_len, activation="softmax"), name="notes_output"
)(out_notes)

out_times = TimeDistributed(Dense(256, activation="relu"))(x)
out_times = TimeDistributed(Dense(128, activation="relu"))(out_times)
out_times = TimeDistributed(Dense(64, activation="relu"))(out_times)
# out_times = TimeDistributed(Dropout(0.5))(out_times)
out_times = TimeDistributed(Dense(2), name="times_output")(out_times)


model = Model(inputs=inputs, outputs=[out_notes, out_times])
metrics = {
    "notes_output": "categorical_accuracy",  # Adjust based on your task
    "times_output": "mae",  # Mean Absolute Error
}
losses = {
    "notes_output": "categorical_crossentropy",  # Adjust based on your task
    "times_output": "mean_squared_error",  # Mean Squared Error
}
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=losses,  # 'mse
    metrics=metrics,
)
# fit model

model.summary()

# plot model
from keras.utils import plot_model

plot_model(model, to_file="model_plot.png", show_shapes=True, show_layer_names=True)

model.fit(X, [y_notes, y_times], epochs=220, verbose=2)

####################test model#######################

x_index = 14

X_test = X[x_index].reshape(
    1, X[x_index].shape[0], X[x_index].shape[1], X[x_index].shape[2]
)

print(X_test.shape)

plt.imshow(X[x_index, :, :, 0])
plt.show()

isTest = True
# demonstrate prediction
yhat = model.predict(X_test, verbose=1)

notes_pred_i = 0
# print(yhat[notes_pred_i].shape)
times_pred_i = 1
# print(yhat[times_pred_i].shape)

# print(yhat[notes_pred_i])

if isTest:
    comp_index = 0
else:
    comp_index = x_index


for i, pred in enumerate(yhat[notes_pred_i][comp_index]):
    pred_index = np.argmax(pred)
    true_index = np.argmax(y_notes[x_index][i])
    if pred_index > 0:
        print("Pred", end="----")
        # print pred and true note
        print(
            list(note_tokenizer.keys())[
                list(note_tokenizer.values()).index(pred_index)
            ],
            end="----",
        )
    if true_index > 0:
        print(f"True", end="----")
        print(
            list(note_tokenizer.keys())[list(note_tokenizer.values()).index(true_index)]
        )


for i, pred in enumerate(yhat[times_pred_i][comp_index]):
    time_in = pred[0]
    time_out = pred[1]

    if time_in > 0:
        print(f"Pred", end="----")
        print(f"{time_in:.2f} - {time_out:.2f}", end=" --- ")
    if time_out > 0:
        print(f"True", end="----")
        print(f"{y_times[x_index][i][0]:.2f} - {y_times[x_index][i][1]:.2f}")
