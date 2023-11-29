import tensorflow as tf
from keras import Input, Model
from keras.layers import Conv1D, MaxPooling1D, Dropout, Dense, Flatten, Activation, BatchNormalization, Add
from SS_prepareTrain import x_train, y_train


def residual_block(x, filters, kernel_size, activation='relu'):
    y = Conv1D(filters=filters, kernel_size=kernel_size,
               activation=activation, padding='same')(x)
    y = BatchNormalization()(y)
    y = Conv1D(filters=filters, kernel_size=kernel_size,
               activation=None, padding='same')(y)
    y = BatchNormalization()(y)
    y = Add()([x, y])
    y = tf.keras.activations.relu(y)
    return y


def conv(filters, maxPool, inputs, kernel):
    x = Conv1D(filters, kernel_size=kernel)(inputs)
    x = Activation("relu")(x)
    x = MaxPooling1D(maxPool)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    return x


def dense(units, inputs):
    x = Dense(units)(inputs)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    return x


def classifier_branch(inputs, num_outs):
    x = conv(128, 2, inputs, 8)
    x = residual_block(x, 128, 8)
    x = conv(128, 2, x, 4)
    x = conv(64, 1, x, 4)
    x = residual_block(x, 64, 4)
    x = Flatten()(x)

    x = dense(128, x)
    x = dense(64, x)
    x = dense(32, x)
    x = Dense(num_outs)(x)
    # actiavion for a binary classification
    output = Activation("sigmoid")(x)

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
model.compile(loss='binary_crossentropy',
              optimizer=optimizer, metrics=['accuracy'])


# Train the model using the x_train and y_train arrays
model.fit(x_train, y_train, batch_size=8, epochs=60, shuffle=False)

# save the model
model.save('stringStruckModelV2.h5')
