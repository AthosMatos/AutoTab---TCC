from keras.models import load_model
import numpy as np
from utils.audio.load_prepare import loadAndPrepare
from utils.paths import CUSTOM_DATASETS, RAW_DATASETS
import librosa
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from utils.audio.rmsNorm import rmsNorm

SR = 44100
CUT_SECS = 4


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


train_ds = np.load("seq2seqNpyIDMT.npz")

GUITAR_NOTES = train_ds["GUITAR_NOTES"]


path = RAW_DATASETS.path + "/musics/g chord.wav"
# path = RAW_DATASETS.path + "/musics/beach house - clean.wav"
# path = RAW_DATASETS.path + "/musics/riffs test.wav"
"""
    riff test notes in riff
    1- A2 E3 A3
    2- D3 A3 D4
    3 - E3 B3 E4
"""

model = load_model("Models/seq1seqModel.h5")
FILEWAV, SR = librosa.load(path, sr=SR)
spec = prepare(FILEWAV)

print(spec.shape)

plt.imshow(spec, aspect="auto", origin="lower")
plt.show()

yhat = model.predict(spec.reshape(1, spec.shape[0], spec.shape[1], 1))

note_tokenizer = Tokenizer(lower=False, filters="")
note_tokenizer.fit_on_texts(GUITAR_NOTES)
note_tokenizer = note_tokenizer.word_index


timePred = yhat[1][0]
for i, pred in enumerate(yhat[0][0]):
    pred_index = np.argmax(pred)
    time_in = timePred[i][0]
    time_out = timePred[i][1]

    if pred_index > 0:
        print("Pred", end=": ")
        # print pred and true note
        print(
            list(note_tokenizer.keys())[
                list(note_tokenizer.values()).index(pred_index)
            ],
            end=" ",
        )
        print(f": {time_in:.2f} - {time_out:.2f}")
