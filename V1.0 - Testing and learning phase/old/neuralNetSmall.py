import tensorflow as tf
from prepareTrain import audio_inputs, notes_outputs, all_notes

kernel_size = 8

# Define the model architecture
model = tf.keras.Sequential([
    # First convolutional layer
    tf.keras.layers.Conv1D(128, kernel_size, activation='relu',  # dilation_rate=2,
                           input_shape=(audio_inputs.shape[1], 44)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Dropout(0.25),

    # Second convolutional layer
    tf.keras.layers.Conv1D(128, kernel_size, activation='relu',  # dilation_rate=2,
                           ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(1),
    tf.keras.layers.Dropout(0.25),
    # Third convolutional layer
    tf.keras.layers.Conv1D(64, kernel_size, activation='relu',  # dilation_rate=2,
                           ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(1),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(),
    # Dense layer
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    # Dense layer
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    # Dense layer
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    # Output layer
    tf.keras.layers.Dense(all_notes.__len__(), activation='softmax')
])
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
# Compile the model with categorical crossentropy loss and accuracy metric
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer, metrics=['accuracy'])

model.summary()

# Train the model using the x_train and y_train arrays
model.fit(audio_inputs, notes_outputs, batch_size=64, epochs=40, shuffle=True)

# save the model
model.save('test_modelSmall.h5')
