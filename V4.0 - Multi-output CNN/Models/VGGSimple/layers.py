import tensorflow as tf


def convLayer(filters, kernel, strides, input_layer):
    x = tf.keras.layers.Conv2D(
        filters, kernel, activation="relu", padding="same", strides=strides
    )(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)

    return x

def denseLayer(units, input_layer):
    x = tf.keras.layers.Dense(units, activation="relu")(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    return x

def inputLayer(input_shape):
    return tf.keras.Input(shape=input_shape)

def outputLayer(num_labels,input_layer):
    return tf.keras.layers.Dense(num_labels, activation="softmax")(input_layer)
    