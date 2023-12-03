import os
from Helpers.consts.paths import guitarset_ds_path
import numpy as np
from keras.utils import to_categorical
from Helpers.utils.notes import genNotes
import jams
import librosa
import soundfile as sf

LABELS = genNotes()[0]


def getFilesPATHS(path: str, extension=".wav"):
    paths = []
    for DIRS, _, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                paths.append(os.path.join(DIRS, file))
    return paths


def loadJams(jams_file_path):
    # Load the JAMS file
    jam = jams.load(jams_file_path)

    # Initialize empty lists for "notes" and "timesteps"
    notes = []
    onsets = []
    # durations = []

    # Get the pitch_contour annotations
    annotations = jam.search(namespace="note_midi")
    buffer = 0.1  # 100 milliseconds

    # For each string
    for ann in annotations:
        # For each note
        for i, obs in enumerate(ann.data):
            # Get the start time and end time of the note

            start_time = max(0, obs.time - buffer)

            if i + 1 < len(ann.data):
                end_time = ann.data[i + 1].time + buffer
                # print(obs.time, ann.data[i + 1].time)
            else:
                break
            # Get the pitch of the note

            pitch = obs.value
            # Get the midi note number

            note = librosa.midi_to_note(pitch)
            # Append the note to the list of notes
            notes.append(note)
            # Append the start and end times to the list of timesteps
            onsets.append((start_time, end_time))
            # Now you can use the start_time and end_time to extract the audio of the note or chord

    onsets = np.array(onsets)
    notes = np.array(notes)
    return notes, onsets


def findFilePath(filename, path, extension=".wav"):
    for root, _, files in os.walk(path):
        for file in files:
            filepath = os.path.join(root, file)
            if filepath.endswith(extension):
                # print(filepath)
                # print(filename)
                if filepath.split("\\")[-1] == filename:
                    return filepath

    return None


wavfilepath = guitarset_ds_path + "\\audio_mono-mic\\00_BN1-129-Eb_solo_mic.wav"
print(wavfilepath)
jamFileName = wavfilepath.split("\\")[-1].split(".")[0].strip("_mic").strip("_mix")
print(jamFileName)
foundJamFile = findFilePath(jamFileName + ".jams", guitarset_ds_path, extension=".jams")
if foundJamFile is None:
    print("No jam file found")
    exit()

notes, onsets = loadJams(foundJamFile)
music, sr = librosa.load(wavfilepath, sr=None, mono=True)

# create folder for the notes
if not os.path.exists("notesFromJam"):
    os.makedirs("notesFromJam")

for i, (start, end) in enumerate(onsets):
    # print(start, end)
    sf.write(
        f"notesFromJam/{notes[i]}.wav",
        music[int(start * sr) : int(end * sr)],
        sr,
        subtype="PCM_16",
    )
