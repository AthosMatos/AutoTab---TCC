import librosa
import numpy as np
from sklearn.preprocessing import minmax_scale
from utils.audio.rmsNorm import rmsNorm
import tensorflow as tf
import sys

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")


def Prepare(audio, sample_rate, expand_dims, audio_pad_sec=2.5):
    if audio.shape[0] < int(audio_pad_sec * sample_rate):
        audio = np.pad(
            audio, (0, int(audio_pad_sec * sample_rate) - audio.shape[0]), "constant"
        )
    elif audio.shape[0] > int(audio_pad_sec * sample_rate):
        audio = audio[0 : int(audio_pad_sec * sample_rate)]

    audio = rmsNorm(audio, -50)

    D = np.abs(
        librosa.cqt(
            y=audio,
            sr=sample_rate,
            fmin=librosa.note_to_hz("C2"),
        )
    )
    D = librosa.amplitude_to_db(D, ref=np.max)
    D = minmax_scale(D)
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
    path, audio_limit_sec=(None, None), sample_rate=None, expand_dims=True
):
    audio, sample_rate = load(path, audio_limit_sec, sample_rate)

    S = Prepare(audio, sample_rate, expand_dims)

    return S, sample_rate
