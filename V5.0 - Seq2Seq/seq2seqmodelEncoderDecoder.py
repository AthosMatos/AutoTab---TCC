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

train_ds = np.load("seq2seqNpyIDMT.npz", allow_pickle=True)

X = train_ds["X"]
y_notes = train_ds["y_notes"]
y_times = train_ds["y_times"]
GUITAR_NOTES = train_ds["GUITAR_NOTES"]

print(X.shape)
print(y_notes.shape)
print(y_times.shape)
max_seq = 64
notes_outs_len = y_notes.shape[2]

note_tokenizer = Tokenizer(lower=False, filters="")
note_tokenizer.fit_on_texts(GUITAR_NOTES)
note_tokenizer = note_tokenizer.word_index


# define mode
inputs = Input(shape=(X.shape[1], X.shape[2], X.shape[3]))


x = Conv2D(32, 3, padding="same")(inputs)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

x = Conv2D(64, 3, padding="same")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

x = Conv2D(128, 3, padding="same")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

convoluted_vector = TimeDistributed(Flatten())(x)

""" encoder_output = LSTM(256, return_sequences=True)(convoluted_vector)

decoder_output = LSTM(128)(encoder_output)
 """
out_notes = TimeDistributed(Dense(128))(convoluted_vector)
out_notes = TimeDistributed(BatchNormalization())(out_notes)
out_notes = TimeDistributed(Activation("relu"))(out_notes)
out_notes = TimeDistributed(
    Dense(notes_outs_len, activation="softmax"), name="notes_output"
)(out_notes)

out_times = TimeDistributed(Dense(128))(convoluted_vector)
out_times = TimeDistributed(BatchNormalization())(out_times)
out_times = TimeDistributed(Activation("relu"))(out_times)
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
    optimizer="rmsprop",  #'rmsprop'
    loss=losses,  # 'mse
    metrics=metrics,
)
# fit model

model.summary()

# plot model
from keras.utils import plot_model

plot_model(model, to_file="EncoderDecoder.png", show_shapes=True, show_layer_names=True)

exit()


def lr_scheduler(epoch, lr):
    if epoch % 40 == 0 and epoch != 0:
        lr = lr * 0.9
    return lr


lr_schedule = LearningRateScheduler(lr_scheduler)
checkpoint_callback = ModelCheckpoint(
    filepath="/content/drive/MyDrive/AutoTAB/Models/bestseq_model.h5",
    save_best_only=True,
    monitor="val_loss",
    mode="min",
    verbose=1,
)

model.fit(X, [y_notes, y_times], epochs=160, verbose=1, callbacks=[])

# save the model
model.save("/content/drive/MyDrive/AutoTAB/Models/seq1seqModel.h5")

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
        print("Pred", end=": ")
        # print pred and true note
        print(
            list(note_tokenizer.keys())[
                list(note_tokenizer.values()).index(pred_index)
            ],
            end="  ",
        )
    if true_index > 0:
        print(f"True", end=": ")
        print(
            list(note_tokenizer.keys())[list(note_tokenizer.values()).index(true_index)]
        )


for i, pred in enumerate(yhat[times_pred_i][comp_index]):
    time_in = pred[0]
    time_out = pred[1]

    print(f"Pred", end="---->")
    print(f"| {time_in:.2f} - {time_out:.2f} |", end="  ")

    print(f"True", end="---->")
    print(f"| {y_times[x_index][i][0]:.2f} - {y_times[x_index][i][1]:.2f} |")
