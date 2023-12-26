import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
from utils.audio.rmsNorm import rmsNorm
import sys
from sklearn.preprocessing import minmax_scale
import tensorflow as tf

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")


def Prepare(audio, sample_rate, expand_dims=True, amp_to_db=True):
    # audio = librosa.effects.harmonic(y=audio, margin=3)  # best so far for chords
    D = np.abs(
        librosa.cqt(
            y=audio,
            sr=sample_rate,
            fmin=librosa.note_to_hz("C2"),  # best so far
        )
    )
    # D = rmsNorm(D, -30)
    if amp_to_db:
        """D = librosa.amplitude_to_db(D, ref=np.max)"""

    D = minmax_scale(D)

    """ if D.shape[1] < 216:
        D = np.pad(D, ((0, 0), (0, 216 - D.shape[1])), "constant", constant_values=0)
    elif D.shape[1] > 216:
        D = D[:, :216] """

    if expand_dims:
        D = np.expand_dims(D, -1)

    D = tf.image.resize(D, (128, 128), method="bicubic").numpy()

    return D


def load(path, seconds_limit=(None, None), sample_rate=None):
    sec_start, sec_end = seconds_limit

    if sec_start and sec_end:
        sec_end = sec_end - sec_start

    audio, sample_rate = librosa.load(
        path,
        mono=True,
        sr=sample_rate,
        offset=sec_start,
        duration=sec_end,
    )

    return audio, sample_rate


def loadAndPrepare(
    path,
    audio_limit_sec=(None, None),
    sample_rate=None,
    expand_dims=True,
    amp_to_db=True,
):
    audio, sample_rate = load(path, audio_limit_sec, sample_rate)

    S = Prepare(audio, sample_rate, expand_dims=expand_dims, amp_to_db=amp_to_db)

    return S, sample_rate


def standardize(S: np.ndarray):
    # Reshape the data to 2D for standardization
    S_2d = S.reshape(-1, S.shape[-1])

    # Standardize the data
    scaler = StandardScaler()
    S_2d_scaled = scaler.fit_transform(S_2d)

    # Reshape the data back to its original shape
    S_scaled = S_2d_scaled.reshape(S.shape)

    return S_scaled
