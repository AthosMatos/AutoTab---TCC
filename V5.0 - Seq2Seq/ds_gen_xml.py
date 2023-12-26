import xmltodict
from utils.paths import RAW_DATASETS, SEQ2SEQ
from utils.files.loadFiles import getFilesPATHS, findFilePath
import librosa
import os
import numpy as np
from utils.notes import genNotes_v2
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt
import soundfile as sf
from utils.audio.rmsNorm import rmsNorm

__DS_NAME__ = "IDMT-SMT-GUITAR_V2"
GUITAR_NOTES = genNotes_v2("F#1", "A6")
SR = 44100
CUT_SECS = 4
XMLFILES = getFilesPATHS(
    RAW_DATASETS.IDMT_SMT_GUITAR_V2,
    ignores=["dataset4"],
    extension=".xml",
    randomize=True,
)

SAVE_PATH = SEQ2SEQ.path
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)


def get_seconds_samples_note(EVENT):
    E_NOTE = librosa.midi_to_note(float(EVENT["pitch"])).replace("♯", "#")

    E_START_SECONDS = float(EVENT["onsetSec"])
    E_END_SECONDS = float(EVENT["offsetSec"])

    E_START_SAMPLES = int((float(EVENT["onsetSec"])) * SR)
    E_END_SAMPLES = int((float(EVENT["offsetSec"])) * SR)

    return (
        E_NOTE,
        E_START_SECONDS,
        E_END_SECONDS,
        E_START_SAMPLES,
        E_END_SAMPLES,
    )


def prepareXML(xml_string):
    AUDIO_DATA = xmltodict.parse(xml_string)["instrumentRecording"]
    FILE_NAME = AUDIO_DATA["globalParameter"]["audioFileName"]
    FILE_NAME_CLEAN = FILE_NAME.split("\\")[-1].split(".")[0]
    print(FILE_NAME_CLEAN)

    AUDIO_EVENTS = AUDIO_DATA["transcription"]["event"]
    # if is a list
    if type(AUDIO_EVENTS) != list:
        AUDIO_EVENTS = [AUDIO_EVENTS]

    return AUDIO_EVENTS, FILE_NAME_CLEAN


# for now ignoring dataset4


def prepare(AUDIO):
    # pad wave tp cutsecs
    if AUDIO.shape[0] < CUT_SECS * SR:
        AUDIO = np.pad(
            AUDIO, (0, (CUT_SECS * SR) - AUDIO.shape[0]), "constant", constant_values=0
        )
    else:
        AUDIO = AUDIO[0 : CUT_SECS * SR]

    AUDIO = rmsNorm(AUDIO, -50)
    AUDIO = librosa.effects.harmonic(AUDIO)

    D = np.abs(librosa.cqt(AUDIO, sr=SR, fmin=librosa.note_to_hz("C2")))
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

for XMLFILE in XMLFILES:
    # FILEXML = findFilePath("LP_Lick6_FN.xml", OLD_DS_PATH)
    # print("FILEXML: ", XMLFILE)
    XMLPATH = XMLFILE.split("\\annotation\\")[0]
    # print("XMLPATH: ", XMLPATH)
    WAVFILENAME = XMLFILE.split("\\")[-1].split(".")[0] + ".wav"
    # print("WAVFILENAME: ", WAVFILENAME)
    WAVFILEPATH = findFilePath(WAVFILENAME, XMLPATH)

    porcentage = (filesRead * 100) / len(XMLFILES)
    print(f"Loaded {filesRead} of {len(XMLFILES)} files ({porcentage:.2f}%)")

    if WAVFILEPATH is None:
        # print(f"File {WAVFILENAME} not found")
        continue
    else:
        print(f"|| {WAVFILENAME} ||")
        print()

    FILEWAV, SR = librosa.load(WAVFILEPATH, sr=SR)

    with open(XMLFILE, "r") as f:
        AUDIO_EVENTS, FILE_NAME_CLEAN = prepareXML(f.read())

        file_name = FILE_NAME_CLEAN
        if not os.path.exists(os.path.join(SAVE_PATH, file_name)):
            os.makedirs(os.path.join(SAVE_PATH, file_name))

        save_p = os.path.join(SAVE_PATH, file_name)

        secs_start = 0
        save_end = CUT_SECS

        y_n = []
        y_t = []

        save_n = []
        save_t = []

        for EvI, event in enumerate(AUDIO_EVENTS):
            (
                curr_note,
                curr_start_seconds,
                curr_end_seconds,
                curr_start_samples,
                curr_end_samples,
            ) = get_seconds_samples_note(event)

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
                    max_seq = y_n.__len__()
                wav = FILEWAV[int(secs_start * SR) : int(save_end * SR)]
                spc = prepare(wav)
                print(spc.shape)
                X.append(spc)
                sf.write(
                    os.path.join(save_p, f"{EvI}-{secs_start}-{save_end}.wav"),
                    wav,
                    SR,
                )
                """ np.savez_compressed(
                    os.path.join(save_p, f"{EvI}-{secs_start}-{save_end}"),
                    y_notes=y_n,
                    y_times=y_t,
                ) """
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
                max_seq = y_n.__len__()

            wav = FILEWAV[int(secs_start * SR) : int(save_end * SR)]
            spc = prepare(wav)
            print(spc.shape)
            X.append(spc)
            sf.write(
                os.path.join(save_p, f"{EvI}-{secs_start}-{save_end}.wav"),
                wav,
                SR,
            )

        filesRead = filesRead + 1
        """ np.savez_compressed(
            os.path.join(save_p, f"{EvI}-{secs_start}-{save_end}"),
            y_notes=y_n,
            y_times=y_t,
        ) """
        """ if len(X) > 10:
            break """
""" fig, axs = plt.subplots(1, len(y_notes) // 3, figsize=(10, 10))

for i, notes in enumerate(y_notes):
    if i % 3 == 0:
        print(X[i].shape, len(notes), len(y_times[i]))
        axs[int(i / 3)].imshow(X[i], aspect="auto", origin="lower")
        axs[int(i / 3)].set_title(notes)

plt.show() """

X = np.array(X)

note_tokenizer = Tokenizer(lower=False, filters="")
note_tokenizer.fit_on_texts(GUITAR_NOTES)
note_tokenizer = note_tokenizer.word_index

max_seq = 43

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

""" # Print the results
print("Tokenized and Padded Note Sequences:")
print(y_)
print("Tokenized and Padded Time Sequences:")
print(y_times_padded) """

print(X.shape)
print(y_.shape)
print(y_times_padded.shape)

# shuffle arrays but keep the same order
p = np.random.permutation(len(X))
X = X[p]
y_ = y_[p]
y_times_padded = y_times_padded[p]

np.savez_compressed(
    "seq2seqNpyIDMT", X=X, y_notes=y_, y_times=y_times_padded, GUITAR_NOTES=GUITAR_NOTES
)
