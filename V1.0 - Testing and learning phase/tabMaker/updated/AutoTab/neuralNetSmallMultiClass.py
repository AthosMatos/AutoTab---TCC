import tensorflow as tf
from keras import Input,Model
from keras.layers import Conv1D, MaxPooling1D, Dropout, Dense, Flatten, Activation, BatchNormalization
from TrainVars import audio_inputs,x_train,y_train, notes_outputs, amps_outputs, playS_outputs, playS2_outputs, gain_outputs

kernel_size = 8

def conv(filters,maxPool, inputs):
    x = Conv1D(filters,kernel_size)(inputs)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(maxPool)(x)
    x = Dropout(0.25)(x) 
    return x

def dense(units,inputs):
    x = Dense(units)(inputs)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    return x

def default_conv_layers(inputs):
    x = conv(128,2,inputs)
    x = conv(128,1,x) 
    x = conv(64,1,x)
    return x

def notes_classifier_branch(inputs, num_outs):
        x = default_conv_layers(inputs)
        x = Flatten()(x)
        x = dense(256,x)
        x = dense(128,x)
        x = dense(64,x)
        x = Dense(num_outs)(x)
        x = Activation("softmax", name="notes_output")(x)

        model = Model(inputs=inputs, outputs=x, name="notes_classifier")
        return model

def gainLevel_classifier_branch(inputs, num_outs):
        x = default_conv_layers(inputs)
        x = Flatten()(x)
        x = dense(128,x)
        x = dense(64,x)
        x = Dense(num_outs)(x)
        x = Activation("softmax", name="gainLevel_output")(x)

        model = Model(inputs=inputs, outputs=x, name="gainLevel_classifier")
        return model

def amplifier_classifier_branch(inputs, num_outs):
        x = default_conv_layers(inputs)
        x = Flatten()(x)
        x = dense(128,x)
        x = dense(64,x)
        x = Dense(num_outs)(x)
        x = Activation("softmax", name="amplifier_output")(x)

        model = Model(inputs=inputs, outputs=x, name="amplifier_classifier")
        return model


def buildModel(summary = False):
    CNN_inputs = Input(shape=(audio_inputs.shape[1], 44))
    notes_class_model = notes_classifier_branch(CNN_inputs, notes_outputs.shape[1])
    notes_output = notes_class_model.output
    gainLevel_class_model = gainLevel_classifier_branch(CNN_inputs, gain_outputs.shape[1])
    gainLevel_output = gainLevel_class_model.output
    amplifier_class_model = amplifier_classifier_branch(CNN_inputs, amps_outputs.shape[1])
    amplifier_output = amplifier_class_model.output
    model = Model(inputs=CNN_inputs, outputs=[notes_output, gainLevel_output, amplifier_output], name="MusicAllNet")

    if summary:
        notes_class_model.summary()
        gainLevel_class_model.summary()
        amplifier_class_model.summary()
        model.summary()
    return model
    

model = buildModel(summary=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
# Compile the model with categorical crossentropy loss and accuracy metric
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer, metrics=['accuracy'])

model.summary()

# Train the model using the x_train and y_train arrays
model.fit(x_train, y_train, batch_size=64, epochs=40, shuffle=True)

# save the model
model.save('test_modelSmallMultiClass.h5')
