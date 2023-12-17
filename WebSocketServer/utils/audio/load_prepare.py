import librosa
import numpy as np
from sklearn.preprocessing import minmax_scale
from utils.audio.rmsNorm import rmsNorm
import tensorflow as tf
import sys


if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")


def Prepare(
    audio,
    sample_rate,
    expand_dims=True,
    transpose=False,
    pad=True,
):
    audio = rmsNorm(audio, -50)

    D = np.abs(
        librosa.cqt(
            y=audio,
            sr=sample_rate,
            fmin=librosa.note_to_hz("C1"),
        )
    )
    D = librosa.amplitude_to_db(D, ref=np.max)
    D = minmax_scale(D)

    if pad:
        if D.shape[1] < 216:
            D = np.pad(
                D, ((0, 0), (0, 216 - D.shape[1])), "constant", constant_values=0
            )
        elif D.shape[1] > 216:
            D = D[:, 0:216]
    if transpose:
        D = D.T
    if expand_dims:
        D = D[..., tf.newaxis]
    return D


def load(path, seconds_limit=(None, None), sample_rate=None):
    audio, sample_rate = librosa.load(path, mono=True, sr=sample_rate)

    if seconds_limit[0] != None and seconds_limit[1] != None:
        audio = audio[
            int(seconds_limit[0] * sample_rate) : int(seconds_limit[1] * sample_rate)
        ]

    return audio, sample_rate


def loadAndPrepare(
    path,
    audio_limit_sec=(None, None),
    sample_rate=None,
    expand_dims=True,
    transpose=False,
    pad=True,
):
    audio, sample_rate = load(path, audio_limit_sec, sample_rate)

    S = Prepare(audio, sample_rate, expand_dims, transpose, pad)

    return S, sample_rate
