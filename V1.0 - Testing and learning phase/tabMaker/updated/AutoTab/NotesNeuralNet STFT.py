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
)
from prepareTrain2 import audio_inputs, notes_outputs

kernel_size = 5


def conv(filters, maxPool, inputs):
    x = Conv1D(filters, kernel_size)(inputs)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(maxPool)(x)
    x = Dropout(0.5)(x)
    return x


def dense(units, inputs):
    x = Dense(units)(inputs)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    return x


def default_conv_layers(inputs):
    x = conv(128, 2, inputs)
    x = conv(64, 2, x)
    return x


def notes_classifier_branch(inputs, num_outs):
    x = default_conv_layers(inputs)
    x = Flatten()(x)
    x = dense(64, x)
    x = dense(32, x)
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


model = buildModel(summary=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(
    loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
)


# Train the model using the x_train and y_train arrays
model.fit(audio_inputs, notes_outputs, batch_size=64, epochs=64, shuffle=True)

# save the model
model.save("NotesNeuralNet_STFT.h5")
