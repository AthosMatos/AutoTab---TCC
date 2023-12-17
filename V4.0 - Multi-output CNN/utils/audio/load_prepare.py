import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
from utils.audio.rmsNorm import rmsNorm
import sys
from sklearn.preprocessing import normalize, minmax_scale
from python_speech_features import mfcc
import tensorflow as tf
import keras

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

""" D = tf.signal.stft(audio, frame_length=255, frame_step=128)
    # Obtain the magnitude of the STFT.
    D = tf.abs(D) """

""" D = librosa.amplitude_to_db(
    np.abs(
        librosa.stft(
            y=audio,
            # n_fft=256,
            # hop_length=256
            # sr=sample_rate,
            # bins_per_octave=12,
            # hop_length=256,
        ),
    ),
    ref=np.max,
).T[:, :] """
""" D = np.minimum(
    D,
    librosa.decompose.nn_filter(D, aggregate=np.median, metric="cosine"),
) """
# D = scipy.ndimage.median_filter(D, size=(1, 9))
""" if D.shape[1] < 216:
    D = np.pad(
        D, ((0, 0), (0, 216 - D.shape[1])), "constant", constant_values=np.min(D)
    )
else:
    D = D[:, 0:216]
# D = D.T
D = minmax_scale(D)
D = np.expand_dims(D.T, -1) """
# D = np.expand_dims(D, -1)
# D = tf.keras.layers.Resizing(32, 32)(D)
# D = minmax_scale(np.squeeze(D, -1))

# print(f"min: {np.min(D)} max: {np.max(D)}")

""" D = mfcc(
    signal=audio,
    samplerate=sample_rate,
    numcep=84,
    nfft=2048,  # 2048
    nfilt=84,
    ceplifter=128,
).T
# minVal = np.min(D)
D = np.expand_dims(D, -1)
D = keras.layers.Resizing(128, 128, "nearest")(D)
D = standardize(np.squeeze(D, -1)) """

""" test identification with padding and with resizing """


def Prepare(
    audio,
    sample_rate,
    expand_dims=True,
    transpose=True,
    pad=True,
):
    # audio = librosa.effects.harmonic(y=audio, margin=18)

    """
    when using this method dont use batch norm in the model
    """

    audio = rmsNorm(audio, -50)
    D = librosa.amplitude_to_db(
        np.abs(
            librosa.cqt(
                y=audio,
                sr=sample_rate,
                fmin=librosa.note_to_hz("C1"),
                n_bins=120,
                bins_per_octave=14,
                # n_fft=256,
                # hop_length=256
                # sr=sample_rate,
                # bins_per_octave=12,
                # hop_length=256,
            )
        ),
        ref=np.max,
    )

    if D.shape[1] < 216:
        D = np.pad(
            D, ((0, 0), (0, 216 - D.shape[1])), "constant", constant_values=np.min(D)
        )
    else:
        D = D[:, 0:216]

    D = minmax_scale(D)
    D = np.expand_dims(D.T, -1)
    # D = standardize(D)
    # D = normalize(D, axis=0)
    # print(f"min: {np.min(D)} max: {np.max(D)}")
    return D


def load(path, seconds_limit=(None, None), sample_rate=None):
    audio, sample_rate = librosa.load(
        path,
        mono=True,
        sr=sample_rate,
        offset=seconds_limit[0],
        duration=seconds_limit[1],
    )

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


def standardize(S: np.ndarray):
    # Reshape the data to 2D for standardization
    S_2d = S.reshape(-1, S.shape[-1])

    # Standardize the data
    scaler = StandardScaler()
    S_2d_scaled = scaler.fit_transform(S_2d)

    # Reshape the data back to its original shape
    S_scaled = S_2d_scaled.reshape(S.shape)

    return S_scaled
