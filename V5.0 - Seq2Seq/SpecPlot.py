import matplotlib.pyplot as plt
import numpy as np
from utils.paths import CUSTOM_DATASETS, RAW_DATASETS
from utils.files.loadFiles import getFilesPATHS, findFilePath
import librosa
from sklearn.preprocessing import minmax_scale
from utils.audio.rmsNorm import rmsNorm

CUT_SECS = 3
SR = 44100
# each cut is 87 samples


def prepare(AUDIO):
    # pad wave tp cutsecs
    """if AUDIO.shape[0] < CUT_SECS * SR:
        AUDIO = np.pad(
            AUDIO, (0, (CUT_SECS * SR) - AUDIO.shape[0]), "constant", constant_values=0
        )
    else:
        AUDIO = AUDIO[0 : CUT_SECS * SR]"""

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
    return np.expand_dims(D, -1)


paths = getFilesPATHS(
    RAW_DATASETS.IDMT_SMT_GUITAR_V2,
    ignores=["dataset4"],
    extension=".wav",
    randomize=True,
    maxFiles=10,
)

specs = []

for path in paths:
    wave = librosa.load(path, sr=SR, duration=CUT_SECS)[0]
    spec = prepare(wave)
    print(spec.shape)
    specs.append(spec)


fig, ax = plt.subplots(nrows=specs.__len__() // 2, ncols=2, figsize=(10, 8))

k = 0
i = 0
j = 0
while i < specs.__len__():
    if i == specs.__len__() // 2:
        k += 1
        j = 0
    if k == 2:
        break
    ax[j][k].imshow(specs[i], aspect="auto", origin="lower")
    ax[j][k].set_title(paths[i].split("\\")[-1])
    i += 1
    j += 1

plt.show()
