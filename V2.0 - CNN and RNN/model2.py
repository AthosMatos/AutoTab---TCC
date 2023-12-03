import tensorflow as tf
from keras import Input, Model
from keras.layers import (
    Conv1D,
    MaxPooling1D,
    Dropout,
    Dense,
    Flatten,
    Activation,
    BatchNormalization,
    Resizing,
    Normalization,
)
from Helpers.DataPreparation.loadDataset import audio_inputs, notes_outputs
from keras.optimizers import Adam

kernel_size = 3


def conv(filters, maxPool, inputs):
    x = Conv1D(filters, kernel_size)(inputs)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(maxPool)(x)
    x = Dropout(0.25)(x)
    return x


def dense(units, inputs):
    x = Dense(units)(inputs)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    return x


def default_conv_layers(inputs):
    x = conv(128, 3, inputs)
    x = conv(128, 2, x)
    x = conv(64, 2, x)

    return x


def notes_classifier_branch(inputs, num_outs):
    norm = Normalization()(inputs)
    x = default_conv_layers(norm)
    x = Flatten()(x)
    x = dense(256, x)
    x = dense(128, x)
    x = Dense(num_outs)(x)
    x = Activation("softmax", name="notes_output")(x)

    model = Model(inputs=inputs, outputs=x, name="notes_classifier")
    return model


def buildModel(summary=False):
    CNN_inputs = Input(shape=(audio_inputs.shape[1], audio_inputs.shape[2]))
    notes_class_model = notes_classifier_branch(CNN_inputs, notes_outputs.shape[1])
    notes_output = notes_class_model.output
    model = Model(inputs=CNN_inputs, outputs=notes_output, name="NotesClassifier")

    if summary:
        model.summary()
    return model


init_lr = 1e-4
EPOCHS = 32
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=init_lr, decay_steps=EPOCHS, decay_rate=0.1
)
opt = Adam(learning_rate=lr_schedule)

model = buildModel(summary=True)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"], loss_weights=[1.5])


# Train the model using the x_train and y_train arrays
model.fit(audio_inputs, notes_outputs, batch_size=32, epochs=EPOCHS)

# save the model
model.save("model2.keras")
