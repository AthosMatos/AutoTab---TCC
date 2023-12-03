import numpy as np
from Helpers.DataPreparation.audioUtils import load, rmsNorm
import os
import librosa
import matplotlib.pyplot as plt


def prep(ad, rms, sr, ax1, ax2, label):
    AUDIO = rmsNorm(ad, rms)
    oenv = librosa.onset.onset_strength(y=AUDIO, sr=sr)
    times = librosa.times_like(oenv)
    onset_raw = librosa.onset.onset_detect(onset_envelope=oenv, backtrack=False, sr=sr)
    S = np.abs(librosa.stft(y=AUDIO))
    rms = librosa.feature.rms(S=S)
    onset_bt = librosa.onset.onset_backtrack(onset_raw, rms[0])

    ax1.set_title(label)

    librosa.display.specshow(
        librosa.amplitude_to_db(S, ref=np.max),
        y_axis="log",
        x_axis="time",
        ax=ax1,
        sr=sr,
    )

    bck_times = librosa.frames_to_time(onset_bt, sr=sr)
    print(bck_times)
    ax2.vlines(
        bck_times,
        0,
        oenv.max(),
        label="Backtracked",
        color="r",
        linewidth=1,
        linestyle="--",
    )
    ax2.legend()


MUSICTESTPATH = os.path.dirname(__file__) + "/dataset/musics/beach house - clean.wav"
ad, sr = load(MUSICTESTPATH, (0, 2))
""" MUSICTESTPATH = (
    os.path.dirname(__file__) + "/dataset/training/AMajor/CHORD - A Major 02.wav"
)
ad, sr = load(MUSICTESTPATH) """

ax = plt.figure(figsize=(10, 4)).subplots(2, 2, sharex=True)
prep(ad, -50, sr, ax[0][0], ax[1][0], "50")
prep(ad, -40, sr, ax[0][1], ax[1][1], "40")

# ax[1].plot(times, oenv, label="Onset strength")
""" ax[1].vlines(
    librosa.frames_to_time(onset_raw),
    0,
    oenv.max(),
    label="Raw onsets",
    color="g",
    linewidth=2,
    # linestyle="--",
) """


plt.show()
