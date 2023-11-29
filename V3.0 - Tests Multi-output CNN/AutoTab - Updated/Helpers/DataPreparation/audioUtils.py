import numpy as np
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler, minmax_scale


def rmsNorm(S, level=-70.0):
    rms_level = level
    # linear rms level and scaling factor
    r = 10 ** (rms_level / 10.0)
    a = np.sqrt((len(S) * r**2) / np.sum(S**2))

    # normalize
    S = S * a

    return S


def Prepare(audio, sample_rate):
    if audio.shape[0] < int(2.5 * sample_rate):
        audio = np.pad(audio, (0, int(2.5 * sample_rate) - audio.shape[0]), "constant")
    elif audio.shape[0] > int(2.5 * sample_rate):
        audio = audio[0 : int(2.5 * sample_rate)]

    audio = rmsNorm(audio, -50)

    audio_cqt_spec = np.abs(
        librosa.cqt(
            y=audio,
            sr=sample_rate,
            fmin=librosa.note_to_hz("C1"),
            n_bins=60 * 2,
            bins_per_octave=12 * 2,
        )
    )
    audio_cqt_spec = librosa.amplitude_to_db(audio_cqt_spec, ref=np.max)
    # audio_cqt_spec = minmax_scale(audio_cqt_spec)  # feature_range=(-1, 1))
    audio_cqt_spec = minmax_scale(audio_cqt_spec)

    S = np.expand_dims(audio_cqt_spec, axis=-1)
    return S


def load(path, seconds_limit=(None, None), sample_rate=None):
    audio, sample_rate = librosa.load(path, mono=True, sr=sample_rate)

    if seconds_limit[0] != None and seconds_limit[1] != None:
        audio = audio[
            int(seconds_limit[0] * sample_rate) : int(seconds_limit[1] * sample_rate)
        ]

    return audio, sample_rate


def loadAndPrepare(path, audio_limit_sec=(None, None), sample_rate=None):
    audio, sample_rate = load(path, audio_limit_sec, sample_rate)

    S = Prepare(audio, sample_rate)

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
