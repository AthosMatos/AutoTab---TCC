import os
from utils.paths import RAW_DATASETS, CUSTOM_DATASETS, SEQ2SEQ
import numpy as np
import jams
import librosa
import soundfile as sf
from utils.files.loadFiles import getFilesPATHS, findFilePath
from utils.notes import genNotes_v2
from sklearn.preprocessing import minmax_scale
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

__DS_NAME__ = "GuitarSet"

FILESJAMS = getFilesPATHS(RAW_DATASETS.GuitarSet, extension=".jams")


GUITAR_NOTES = genNotes_v2("F#1", "A6")
SR = 16000
CUT_SECS = 1
JAMFILES = getFilesPATHS(
    RAW_DATASETS.GuitarSet,
    ignores=[],
    extension=".jams",
)

SAVE_PATH = SEQ2SEQ.path
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)


def loadJams(jams_file_path):
    # Load the JAMS file
    jam = jams.load(jams_file_path)
    notes = []
    onsets = []

    annotations = jam.search(namespace="note_midi")
    for anotation in annotations:
        for i, anotation.data[i] in enumerate(anotation.data):
            start_time = anotation.data[i].time
            end_time = anotation.data[i].time + anotation.data[i].duration
            pitch = anotation.data[i].value
            note = librosa.midi_to_note(pitch)
            # Append the note to the list of notes
            notes.append(note)
            # Append the start and end times to the list of timesteps
            onsets.append((start_time, end_time))
            # Now you can use the start_time and end_time to extract the audio of the note or chord

    onsets = np.array(onsets)
    notes = np.array(notes)
    return notes, onsets


FileIndex = 0


def addFileIndex():
    global FileIndex
    FileIndex += 1


def prepareJams(jams_file_path):
    jam = jams.load(jams_file_path)
    audio_ev = []

    def fillAudioEvents():
        annotations = jam.search(namespace="note_midi")
        for anotation in annotations:
            for i in range(0, len(anotation.data) - 1):
                curr_start_time = anotation.data[i].time
                curr_end_time = anotation.data[i].time + anotation.data[i].duration
                curr_pitch = anotation.data[i].value
                curr_note = librosa.midi_to_note(curr_pitch).replace("♯", "#")
                audio_ev.append(
                    {
                        "note": curr_note,
                        "start_time": curr_start_time,
                        "end_time": curr_end_time,
                    }
                )

    fillAudioEvents()
    # order it by start_time
    audio_ev = sorted(audio_ev, key=lambda k: k["start_time"])
    """ print("WAVFILENAME: ", WAVFILENAME)
    for i, ev in enumerate(audio_ev):
        print(f"{i} - {ev}") """

    return audio_ev


def prepare(AUDIO):
    # pad wave tp cutsecs
    if AUDIO.shape[0] < CUT_SECS * SR:
        AUDIO = np.pad(
            AUDIO, (0, (CUT_SECS * SR) - AUDIO.shape[0]), "constant", constant_values=0
        )
    else:
        AUDIO = AUDIO[0 : CUT_SECS * SR]

    D = np.abs(librosa.cqt(AUDIO, sr=SR))
    D = librosa.amplitude_to_db(D, ref=np.max)

    """ if D.shape[1] < 126:
        D = np.pad(
            D, ((0, 0), (0, 126 - D.shape[1])), "constant", constant_values=np.min(D)
        )
    else:
        D = D[:, 0:126] """
    D = minmax_scale(D)
    return np.expand_dims(D.T, -1)


filesRead = 0
X = []
y_nts = []
y_tms = []
y_notes = []
y_times = []
max_seq = 0


for JAMFILE in FILESJAMS:
    # print("JAMFILE: ", JAMFILE)
    WAVFILENAME = JAMFILE.split("\\")[-1].split(".")[0] + ".wav"
    FILE_NAME_CLEAN = WAVFILENAME.split(".")[0]
    # print("JAMPATH: ", WAVFILENAME)
    # print porcentage of files loaded
    porcentage = (filesRead * 100) / len(FILESJAMS)
    print(f"Loaded {filesRead} of {len(FILESJAMS)} files ({porcentage:.2f}%)")
    WAVFILEPATH = findFilePath(
        WAVFILENAME, RAW_DATASETS.GuitarSet, pathCustomEnd=["_mix", "_mic"]
    )
    WAVFILEPATH_MIC, WAVFILEPATH_MIX = WAVFILEPATH
    if WAVFILEPATH is None:
        # print(f"File {WAVFILENAME} not found")
        continue

    for WAVFP in WAVFILEPATH:
        FILEWAV, SR = librosa.load(WAVFP)
        AUDIO_EVENTS = prepareJams(JAMFILE)

        if not os.path.exists(os.path.join(SAVE_PATH, FILE_NAME_CLEAN)):
            os.makedirs(os.path.join(SAVE_PATH, FILE_NAME_CLEAN))

        save_p = os.path.join(SAVE_PATH, FILE_NAME_CLEAN)

        secs_start = 0
        save_end = CUT_SECS

        y_n = []
        y_t = []

        save_n = []
        save_t = []

        for EvI, event in enumerate(AUDIO_EVENTS):
            curr_start_seconds = event["start_time"]
            curr_end_seconds = event["end_time"]
            curr_note = event["note"]

            if curr_start_seconds < save_end:
                y_n.append(curr_note)
                end = round(curr_end_seconds - secs_start, 2)
                start = round(curr_start_seconds - secs_start, 2)
                if start < 0:
                    start = -1
                if end > CUT_SECS:
                    end = 5
                y_t.append(
                    [
                        start,
                        end,
                    ]
                )

                if curr_end_seconds > save_end:
                    save_n.append(curr_note)

                    end = round(curr_end_seconds - (secs_start + CUT_SECS), 2)
                    start = round(curr_start_seconds - (secs_start + CUT_SECS), 2)
                    if start < 0:
                        start = -1
                    if end > CUT_SECS:
                        end = 5
                    save_t.append(
                        [
                            start,
                            end,
                        ]
                    )

            else:
                y_notes.append(y_n)
                y_times.append(y_t)
                if y_n.__len__() > max_seq:
                    max_seq = y_n.__len__() + 1

                wav = FILEWAV[int(secs_start * SR) : int(save_end * SR)]
                X.append(prepare(wav))
                sf.write(
                    os.path.join(save_p, f"{EvI}-{secs_start}-{save_end}.wav"),
                    wav,
                    SR,
                )
                np.savez_compressed(
                    os.path.join(save_p, f"{EvI}-{secs_start}-{save_end}"),
                    y_notes=y_n,
                    y_times=y_t,
                )
                y_n = []
                y_t = []
                y_n.append(curr_note)
                end = round(curr_end_seconds - (secs_start + CUT_SECS), 2)
                start = round(curr_start_seconds - (secs_start + CUT_SECS), 2)
                if start < 0:
                    start = (
                        -1
                    )  # start -1 means that the audio comes from a already play state
                if end > CUT_SECS:
                    end = 5  # end =5 means that audio continues
                y_t.append(
                    [
                        start,
                        end,
                    ]
                )

                y_n.extend(save_n)
                y_t.extend(save_t)
                save_n = []
                save_t = []
                save_end += CUT_SECS
                secs_start += CUT_SECS

            """ print(
                curr_note,
                curr_start_seconds,
                curr_end_seconds,
            ) """

        if y_n.__len__() > 0:
            y_notes.append(y_n)
            y_times.append(y_t)
            if y_n.__len__() > max_seq:
                max_seq = y_n.__len__() + 1
            wav = FILEWAV[int(secs_start * SR) : int(save_end * SR)]
            X.append(prepare(wav))
            sf.write(
                os.path.join(save_p, f"{EvI}-{secs_start}-{save_end}.wav"),
                wav,
                SR,
            )
            # save y_n and y_t
            np.savez_compressed(
                os.path.join(save_p, f"{EvI}-{secs_start}-{save_end}"),
                y_notes=y_n,
                y_times=y_t,
            )

    filesRead = filesRead + 1


X = np.array(X)

note_tokenizer = Tokenizer(lower=False, filters="")
note_tokenizer.fit_on_texts(GUITAR_NOTES)
note_tokenizer = note_tokenizer.word_index

y_ = []
for notes in y_notes:
    y_notes_tokenized = []
    for note in notes:
        y_notes_tokenized.append(note_tokenizer[note])

    y_notes_padded = pad_sequences([y_notes_tokenized], maxlen=max_seq, padding="post")
    # Convert to numpy arrays
    y_notes_padded = np.array(np.squeeze(y_notes_padded, axis=0))
    # covert to categorical
    y_notes_padded = to_categorical(y_notes_padded, num_classes=len(note_tokenizer) + 1)

    y_.append(y_notes_padded)
y_ = np.array(y_)

padValue = -2  # means no value -1 means audio continued from before

for times in y_times:
    for i in range(max_seq - len(times)):
        times.append([padValue, padValue])

y_times_padded = np.array(y_times)

from utils.nameMapping import printTimes, printNotesCategorical

# Print the results
print("Tokenized and Padded Note Sequences:")
printNotesCategorical(y_)
print("Tokenized and Padded Time Sequences:")
printTimes(y_times_padded)

print(X.shape)
print(y_.shape)
print(y_times_padded.shape)

# shuffle arrays but keep the same order
p = np.random.permutation(len(X))
X = X[p]
y_ = y_[p]
y_times_padded = y_times_padded[p]

np.savez_compressed(
    "seq2seqNpyGuitarSet",
    X=X,
    y_notes=y_,
    y_times=y_times_padded,
    GUITAR_NOTES=GUITAR_NOTES,
)
