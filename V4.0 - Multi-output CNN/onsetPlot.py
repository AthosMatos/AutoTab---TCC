import numpy as np
from utils.audio.load_prepare import load
from utils.audio.rmsNorm import rmsNorm
import os
import librosa
import matplotlib.pyplot as plt
from utils.paths import RAW_DATASETS
from sklearn.preprocessing import minmax_scale

fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True)
MUSICTESTPATH = (
    RAW_DATASETS.path + "/musics/beach house - clean.wav"
)  # "/dataset/musics/beach house - clean.wav"
ad, sr = load(MUSICTESTPATH, seconds_limit=(0, 5))


# amptodb = "pre" or "post" or False but initialy False
def getOnsets(
    ad, sr, rmsStrength=-50, amptodb=False, standardize_audio=False, minMaxScale=False
):
    ad = rmsNorm(ad, rmsStrength)
    audio_cqt_spec = np.abs(
        librosa.cqt(
            y=ad,
            sr=sr,
            fmin=librosa.note_to_hz("C2"),
        )
    )

    print("CQT Shape: ", audio_cqt_spec.shape)

    if amptodb == "pre":
        audio_cqt_spec = librosa.amplitude_to_db(audio_cqt_spec, ref=np.max)
    if minMaxScale == "pre":
        audio_cqt_spec = minmax_scale(audio_cqt_spec)

    # AUDIO = ad
    onset_strenght = librosa.onset.onset_strength(S=audio_cqt_spec, sr=sr)
    onset_times = librosa.times_like(onset_strenght, sr=sr)
    onset_raw = librosa.onset.onset_detect(
        onset_envelope=onset_strenght, backtrack=False, sr=sr
    )
    """ rms = librosa.feature.rms(S=S)
    onset_bt = librosa.onset.onset_backtrack(onset_raw, rms[0]) """
    if amptodb == "post":
        audio_cqt_spec = librosa.amplitude_to_db(audio_cqt_spec, ref=np.max)
    if minMaxScale == "post":
        audio_cqt_spec = minmax_scale(audio_cqt_spec)

    return audio_cqt_spec, onset_strenght, onset_times, onset_raw


def plotSpecOnsets(S2, sr, axis, oenv, times, onset_raw, label=""):
    ax[0][axis].set_title(label)

    librosa.display.specshow(
        S2,
        y_axis="cqt_hz",
        x_axis="time",
        sr=sr,
        ax=ax[0][axis],
    )
    ax[0][axis].label_outer()
    ax[1][axis].plot(times, oenv, label="Onset strength")
    ax[1][axis].vlines(
        librosa.frames_to_time(onset_raw, sr=sr),
        0,
        oenv.max(),
        label="Raw onsets",
        color="g",
        linewidth=2,
        # linestyle="--",
    )
    """ ax[1].vlines(
        librosa.frames_to_time(onset_bt),
        0,
        oenv.max(),
        label="Backtracked",
        color="r",
        linewidth=1,
        linestyle="--",
    ) """
    ax[1][axis].legend()
    ax[1][axis].label_outer()


S, oenv, times, onset_raw = getOnsets(ad, sr)
S2, oenv2, times2, onset_raw2 = getOnsets(ad, sr, amptodb="pre", minMaxScale="pre")

plotSpecOnsets(S, sr, 0, oenv, times, onset_raw, "onsets without amplitude to db")
plotSpecOnsets(
    S2, sr, 1, oenv2, times2, onset_raw2, "onsets with amplitude to db and minmaxscale"
)

plt.show()
