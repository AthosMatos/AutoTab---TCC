import jams
from paths import DSPATH
import numpy as np
import librosa
import os
from audioUtils import loadAndPrepare
from keras.utils import to_categorical
from notes import genNotes

path = DSPATH + "GuitarSet"
LABELS = genNotes()[0]
MAXNOTES = 50
MAXTIMESTEPS = 50
TESTAMOUNT = 4
EPOCHS = 50
# print(LABELS)

xTrain = []
notesOut = []
onsetsOut = []


def getFilesAmount(path, extension=".wav"):
    amount = 0
    for _, _, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                amount += 1

    return amount


""" def loadJams(jams_file_path):
    # Load the JAMS file
    jam = jams.load(jams_file_path)

    # Initialize empty lists for "notes" and "timesteps"
    notes = []
    onsets = []
    # durations = []

    # Extract note annotations
    for annotation in jam.annotations:
        if annotation.namespace == "note_midi":
            for note in annotation.data:
                onset_time = note.time
                midi_pitch = librosa.midi_to_note(note.value)
                # duration = note.duration

                # Here, you can map the note to the corresponding time step
                # You might need to convert onset_time to the appropriate time step units
                # For example, if your audio is sampled at 44.1 kHz, onset_time * 44100 could give you the corresponding sample index.

                #print("onset_time: ", onset_time)
                #print("midi_pitch: ", midi_pitch)
                #print("duration: ", duration) 

                notes.append(midi_pitch)
                onsets.append(onset_time)
                # durations.append(duration)

    # order the notes by onset time
    notes = np.array(notes)
    onsets = np.array(onsets)
    # durations = np.array(durations)
    idx = np.argsort(onsets)
    notes = notes[idx]
    onsets = onsets[idx]

    lbls = [v for v, _ in enumerate(LABELS)]
    hot = to_categorical(lbls, num_classes=len(LABELS))

    # print(hot.shape)
    nts = []
    for note in notes:
        nts.append(hot[LABELS.index(note)])

    nts = np.array(nts)
    # durations = durations[idx]
    # print the notes
    #print("onsets: ", onsets.shape)
    #print("notes: ", nts.shape)

    return nts, onsets """


def loadJams(jams_file_path, TIMESTEPS):
    # Load the JAMS file
    jam = jams.load(jams_file_path)

    # Initialize empty lists for "notes" and "timesteps"
    notes = []
    onsets = []
    # durations = []

    # Extract note annotations
    for annotation in jam.annotations:
        if annotation.namespace == "note_midi":
            for note in annotation.data:
                onset_time = note.time
                midi_pitch = librosa.midi_to_note(note.value)

                notes.append(midi_pitch)
                onsets.append(onset_time)

    notes = np.array(notes)
    onsets = np.array(onsets)

    idx = np.argsort(onsets)
    notes = notes[idx]
    onsets = onsets[idx]

    lbls = [v for v, _ in enumerate(LABELS)]
    hot = to_categorical(lbls, num_classes=len(LABELS))

    timestepcap = 2
    nts = []
    ns = []

    for i, note in enumerate(notes):
        if onsets[i] <= timestepcap:
            nts.append(hot[LABELS.index(note)])
        else:
            if len(nts) < MAXNOTES:
                for i in range(MAXNOTES - len(nts)):
                    nts.append(hot[-1])

            nts = np.array(nts)
            # print(nts.shape)
            ns.append(nts)
            nts = []
            timestepcap += 2
            nts.append(hot[LABELS.index(note)])

    nts = []
    if len(ns) < TIMESTEPS:
        for i in range(TIMESTEPS - len(ns)):
            for i in range(MAXNOTES):
                nts.append(hot[-1])
            ns.append(nts)
            nts = []

    ns = np.array(ns)
    print("ns shape", ns.shape)
    return ns, onsets


def findFilePath(filename, path):
    for root, _, files in os.walk(path):
        for file in files:
            filepath = os.path.join(root, file)
            if filepath.split("\\")[-1] == filename:
                return filepath

    return None


def loadDS():
    filesAmount = getFilesAmount(path)
    filesGoneThrough = 0

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".wav"):
                filepath = os.path.join(root, file)
                jamFileName = (
                    filepath.split("\\")[-1].split(".")[0].strip("_mic").strip("_mix")
                )
                # print(jamFileName)
                spec, _ = loadAndPrepare(filepath, MAXTIMESTEPS=MAXTIMESTEPS)
                # print(spec.max())
                """ for spe in spec:
                    print(spe[0]) """

                xTrain.append(spec)

                # print("INput shape", spec.shape)
                notes, onsets = loadJams(
                    findFilePath(jamFileName + ".jams", path), MAXTIMESTEPS
                )
                # print(notes.shape)
                notesOut.append(notes)
                onsetsOut.append(onsets)

                print("{} / {}".format(filesGoneThrough, filesAmount))

                filesGoneThrough += 1
                if filesGoneThrough >= TESTAMOUNT:
                    return


loadDS()
xTrain = np.array(xTrain)
notesOut = np.array(notesOut)
print(xTrain.shape)
print(notesOut.shape)

import numpy as np
from keras.models import Sequential, Model
from keras.layers import (
    LSTM,
    Dense,
    Input,
    Flatten,
    Reshape,
    Conv2D,
    MaxPooling2D,
    TimeDistributed,
    Dropout,
    BatchNormalization,
)
from keras.optimizers import Adam
import tensorflow as tf

input_shape = (MAXTIMESTEPS, xTrain.shape[2], xTrain.shape[3], xTrain.shape[4])


# Create a neural network
model = Sequential(
    [
        Input(shape=input_shape),
        # convolutional layers
        TimeDistributed(Flatten()),
        # dense layers
        LSTM(64, return_sequences=True, recurrent_activation="tanh"),
        TimeDistributed(Dense(MAXNOTES * len(LABELS), activation="relu")),
        TimeDistributed(Reshape((MAXNOTES, len(LABELS)))),
        TimeDistributed(Dense(len(LABELS), activation="softmax")),
    ]
)

# Compile the model
model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])

# build the model
model.build(
    input_shape=(None, input_shape[0], input_shape[1], input_shape[2], input_shape[3])
)

model.summary()

# Train the model with your data.
# Assuming you have train_input, train_target, val_input, and val_target

model.fit(xTrain, notesOut, epochs=EPOCHS, batch_size=64, verbose=2)


# simple prediction test - xtrain shape = (4, 50, 256, 32, 1)
print("Prediction test")
predTestX = np.array([xTrain[0]])
pred = model.predict(predTestX, verbose=0)

print(pred.shape)
""" 
for i, pd in enumerate(pred[0][0]):
    #print(pd)
    print(LABELS[np.argmax(pd)], LABELS[np.argmax(notesOut[0][0][i])]) """


for i, pd in enumerate(pred[0][0]):
    """print(pd)"""
    pred_confidence = np.max(pd)
    pred_confidence = pred_confidence * 100
    """print(pd)"""
    print(pred_confidence, LABELS[np.argmax(pd)], LABELS[np.argmax(notesOut[0][0][i])])
