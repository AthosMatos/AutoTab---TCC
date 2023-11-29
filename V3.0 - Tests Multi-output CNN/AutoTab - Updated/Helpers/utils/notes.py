import numpy as np

# every note in a electric guitar with 22 frets
# Define the standard tuning notes for each string
STD_TUNNING = ["E2", "A2", "D3", "G3", "B3", "E4"]
NOTES = [
    "C",
    "C♯",
    "D",
    "D♯",
    "E",
    "F",
    "F♯",
    "G",
    "G♯",
    "A",
    "A♯",
    "B",
]  # 12 notes in a chromatic scale
# Define the number of frets
STD_FRETS = 22


def indexLabels(labels: list):
    return np.unique(labels, return_inverse=True)[1]


def genNotes(TUNNING=STD_TUNNING, FRETS=STD_FRETS, indexes=False):
    # Create a 2D array to store the notes with octaves
    notes = []
    string_notes = []
    for string in TUNNING:
        octave = int(string[-1])
        note = string[:-1]
        # print(note)
        note_index = NOTES.index(note)
        # print(note)
        # print(note_index)
        strs_notes = []
        for _ in range(FRETS):
            """print(NOTES[note_index])"""
            NOTE = NOTES[note_index]
            notes.append(NOTE + str(octave))
            strs_notes.append(NOTE + str(octave))
            note_index += 1
            # print(NOTE)
            if NOTE == "B":
                octave += 1
                note_index = 0
        string_notes.append(strs_notes)
    notes.append("none")  # Add a rest note
    notes = np.unique(notes)

    if indexes:
        return notes, string_notes, indexLabels(notes)
    return notes, string_notes
