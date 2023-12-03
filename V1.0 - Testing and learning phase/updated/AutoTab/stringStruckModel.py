import tensorflow as tf
from keras import Input, Model
from keras.layers import Conv1D, MaxPooling1D, Dropout, Dense, Flatten, Activation, BatchNormalization
from SS_prepareTrain import x_train, y_train

kernel_size = 2


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


def classifier_branch(inputs, num_outs):
    x = conv(128, 1, inputs)
    x = conv(64, 1, x)
    x = conv(32, 1, x)
    x = Flatten()(x)

    x = dense(128, x)
    x = dense(64, x)
    x = dense(32, x)
    x = Dense(num_outs)(x)
    # actiavion for a binary classification
    output = Activation("softmax")(x)

    model = Model(inputs=inputs, outputs=output)
    return model


def buildModel(summary=False):
    CNN_inputs = Input(shape=(x_train.shape[1], 44))
    class_model = classifier_branch(
        CNN_inputs, y_train.shape[1])
    output = class_model.output
    model = Model(inputs=CNN_inputs, outputs=output,
                  name="stringStruckClassifier")

    if summary:
        model.summary()
    return model


model = buildModel(summary=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer, metrics=['accuracy'])


# Train the model using the x_train and y_train arrays
model.fit(x_train, y_train, batch_size=32, epochs=60)

# save the model
model.save('stringStruckModel.h5')
