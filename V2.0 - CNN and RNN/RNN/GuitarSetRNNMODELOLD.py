# Create a neural network
model = Sequential()
model.add(TimeDistributed(Flatten(input_shape=(MAXTIMESTEPS, 256, 32, 1))))
model.add(TimeDistributed(Dense(128, activation="relu")))
# Add layers to your network as needed
model.add(LSTM(128, return_sequences=True))
model.add(TimeDistributed(Dense(128, activation="relu")))
model.add(TimeDistributed(Dense(MAXNOTES * len(LABELS), activation="softmax")))
model.add(TimeDistributed(Reshape((MAXNOTES, len(LABELS)))))


# Compile the model
model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])

# build the model
model.build(input_shape=(None, MAXTIMESTEPS, 256, 32, 1))

model.summary()
