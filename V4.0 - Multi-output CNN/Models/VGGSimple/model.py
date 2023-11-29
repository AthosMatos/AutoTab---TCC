import tensorflow as tf
import numpy as np
from autotabModels.VGGSimple.layers import (
    convLayer,
    denseLayer,
    inputLayer,
    outputLayer,
)
from DS.load_np_ds import train_ds, val_ds, TRAIN_Y, VAL_Y, INPUT_SHAPE, NUM_LABELS

""" norm_layer = tf.keras.layers.Normalization()
norm_layer.adapt(data=train_ds["x"]) """


def buildModel(summary=True, savePlot=True, fit=True, saveModel=True):
    input_layer = inputLayer(INPUT_SHAPE)
    # x = tf.keras.layers.Resizing(128, 128)(input_layer)
    # x = norm_layer(input_layer)
    x = convLayer(32, 3, 2, input_layer)
    x = convLayer(64, 3, 1, x)
    x = convLayer(128, 3, 1, x)
    x = tf.keras.layers.Flatten()(x)
    x = denseLayer(256, x)
    x = denseLayer(128, x)
    # 6 outputs
    out1 = outputLayer(NUM_LABELS, x)
    out2 = outputLayer(NUM_LABELS, x)
    out3 = outputLayer(NUM_LABELS, x)
    out4 = outputLayer(NUM_LABELS, x)
    out5 = outputLayer(NUM_LABELS, x)
    out6 = outputLayer(NUM_LABELS, x)

    model = tf.keras.Model(
        inputs=input_layer,
        outputs=[
            out1,
            out2,
            out3,
            out4,
            out5,
            out6,
        ],
    )
    if summary:
        model.summary()
    if savePlot:
        tf.keras.utils.plot_model(model, show_shapes=True)

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    if fit:
        model.fit(
            train_ds["x"],
            TRAIN_Y,
            validation_data=(
                val_ds["x"],
                VAL_Y,
            ),
            batch_size=128,
            epochs=64,
            verbose=2
            # callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
        )

    if saveModel:
        # save the model
        model.save("VGG-Simple.h5")
