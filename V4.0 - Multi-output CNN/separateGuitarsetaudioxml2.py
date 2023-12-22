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

__DS_NAME__ = "IDMT-SMT-GUITAR_V2"
GUITAR_NOTES = genNotes_v2("F#1", "A6")
SR = 16000
XMLFILES = getFilesPATHS(
    RAW_DATASETS.IDMT_SMT_GUITAR_V2,
    ignores=["dataset1", "dataset2", "dataset4"],
    extension=".xml",
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


def prepareSubFoldersString():
    # split the wav file path from the DS_NAME foward
    subF = WAVFILEPATH.split(__DS_NAME__ + "\\")[-1].split("\\")
    filename = subF[-1]
    subF = subF[:-2]
    sub = ""
    for i, s in enumerate(subF):
        if i == 0:
            sub = s + "\\"
        else:
            sub += s + "\\"

    # print("filename: ", filename)
    # print("sub: ", sub)
    return sub


# for now ignoring dataset4


def prepare(AUDIO):
    D = np.abs(librosa.cqt(AUDIO, sr=SR))
    D = librosa.amplitude_to_db(D, ref=np.max)

    if D.shape[1] < 126:
        D = np.pad(
            D, ((0, 0), (0, 126 - D.shape[1])), "constant", constant_values=np.min(D)
        )
    else:
        D = D[:, 0:126]
    D = minmax_scale(D)
    return np.expand_dims(D.T, -1)


X = []
y_nts = []
y_tms = []

for XMLFILE in XMLFILES:
    # FILEXML = findFilePath("LP_Lick6_FN.xml", OLD_DS_PATH)
    # print("FILEXML: ", XMLFILE)
    XMLPATH = XMLFILE.split("\\annotation\\")[0]
    # print("XMLPATH: ", XMLPATH)
    WAVFILENAME = XMLFILE.split("\\")[-1].split(".")[0] + ".wav"
    # print("WAVFILENAME: ", WAVFILENAME)
    WAVFILEPATH = findFilePath(WAVFILENAME, XMLPATH)

    if WAVFILEPATH is None:
        # print(f"File {WAVFILENAME} not found")
        continue
    else:
        print(f"|| {WAVFILENAME} ||")
        print()

    FILEWAV, SR = librosa.load(WAVFILEPATH, sr=SR)

    with open(XMLFILE, "r") as f:
        AUDIO_EVENTS, FILE_NAME_CLEAN = prepareXML(f.read())

        y_notes = []
        y_times = []

        file_name = "np_" + FILE_NAME_CLEAN
        if not os.path.exists(os.path.join(SAVE_PATH, file_name)):
            os.makedirs(os.path.join(SAVE_PATH, file_name))

        save_p = os.path.join(SAVE_PATH, file_name)
        curr_secs = 0
        save_start = 0
        save_end = 4

        for event in AUDIO_EVENTS:
            (
                curr_note,
                curr_start_seconds,
                curr_end_seconds,
                curr_start_samples,
                curr_end_samples,
            ) = get_seconds_samples_note(event)

            if curr_secs < 4:
                # print(curr_note, curr_start_seconds, curr_end_seconds)
                y_notes.append(curr_note)
                y_times.append([curr_start_seconds, curr_end_seconds])
                save_end = curr_start_seconds
            else:
                if save_start == save_end:
                    continue

                X.append(
                    prepare(
                        FILEWAV[int(save_start * SR) : int(save_end * SR)],
                    )
                )
                y_nts.append(y_notes)
                y_tms.append(y_times)
                y_notes = []
                y_times = []
                curr_secs = 0
                save_start = curr_start_seconds
                save_end = save_start + 4
            curr_secs += curr_end_seconds - curr_start_seconds

        if y_notes.__len__() > 0 and save_start != save_end:
            X.append(
                prepare(
                    FILEWAV[int(save_start * SR) : int(save_end * SR)],
                )
            )
            y_nts.append(y_notes)
            y_tms.append(y_times)

        notes_outs_len = len(GUITAR_NOTES) + 1


X = np.array(X)

max_seq = 22  # y_times.shape[1]
note_tokenizer = Tokenizer(lower=False, filters="")
note_tokenizer.fit_on_texts(GUITAR_NOTES)
note_tokenizer = note_tokenizer.word_index

y_ = []
for notes in y_nts:
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

for times in y_tms:
    for i in range(max_seq - len(times)):
        times.append([-1, -1])

y_times_padded = np.array(y_tms)

""" # Print the results
print("Tokenized and Padded Note Sequences:")
print(y_)
print("Tokenized and Padded Time Sequences:")
print(y_times_padded) """

print(X.shape)
print(y_.shape)
print(y_times_padded.shape)
save_path = os.path.join(
    f"{save_start} - {save_end}.npz",
)
np.savez_compressed(save_path, X=X, y_notes=y_, y_times=y_times_padded)

# y_notes = np.array(y_notes)
# y_times = np.array(y_times)

# print("X: ", X.shape)
# print("y_notes: ", y_notes.shape)
# print("y_times: ", y_times.shape)
