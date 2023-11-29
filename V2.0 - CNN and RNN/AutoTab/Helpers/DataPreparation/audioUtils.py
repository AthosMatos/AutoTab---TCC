import numpy as np
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler


def rmsNorm(S, level=-70.0):
    rms_level = level
    # linear rms level and scaling factor
    r = 10 ** (rms_level / 10.0)
    a = np.sqrt((len(S) * r**2) / np.sum(S**2))

    # normalize
    S = S * a

    return S


def Prepare(audio, sample_rate):
    # S = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    # S = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128, fmax=8000)
    # S = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=128)
    """S = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128, fmax=2000)
    S = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=128)"""
    # S = np.abs(librosa.stft(audio))

    S = np.abs(librosa.stft(y=audio, win_length=1024, hop_length=1024))[
        :256, :
    ]  # limit to 2000hz max a that a guitar can play

    """ 
    S = librosa.amplitude_to_db(S, ref=np.max) """

    # LIMIT = 216
    LIMIT = 108
    if S.shape[1] < LIMIT:
        S = np.pad(S, ((0, 0), (0, LIMIT - S.shape[1])), "constant")
    elif S.shape[1] > LIMIT:
        S = S[:, 0:LIMIT]

    S = standardize(S)
    """  # add new axis to the end of the array
    S = np.expand_dims(S, axis=-1) """

    return S, sample_rate


def load(path, seconds_limit=(None, None)):
    audio, sample_rate = librosa.load(path, mono=True, sr=None)

    if seconds_limit[0] != None and seconds_limit[1] != None:
        audio = audio[
            int(seconds_limit[0] * sample_rate) : int(seconds_limit[1] * sample_rate)
        ]

    """ if len(audio) < sample_rate * 2.5:
        audio = np.pad(
            audio,
            (0, int(sample_rate * 2.5 - len(audio))),
            "constant",
            constant_values=0,
        )
    else:
        audio = audio[0 : int(sample_rate * 2.5)] """

    return audio, sample_rate


def loadAndPrepare(path, audio_limit_sec=(None, None)):
    audio, sample_rate = load(path, audio_limit_sec)

    S, _ = Prepare(audio, sample_rate)

    return S, sample_rate


def standardize(S):
    # Reshape the data to 2D for standardization
    S_2d = S.reshape(-1, S.shape[-1])

    # Standardize the data
    scaler = StandardScaler()
    S_2d_scaled = scaler.fit_transform(S_2d)

    # Reshape the data back to its original shape
    S_scaled = S_2d_scaled.reshape(S.shape)

    return S_scaled
