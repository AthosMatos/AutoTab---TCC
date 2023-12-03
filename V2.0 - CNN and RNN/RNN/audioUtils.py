import librosa
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.preprocessing import minmax_scale


def rmsNorm(S, level=-70.0):
    rms_level = level
    # linear rms level and scaling factor
    r = 10 ** (rms_level / 10.0)
    a = np.sqrt((len(S) * r**2) / np.sum(S**2))

    # normalize
    S = S * a

    return S


def Prepare(audio, sample_rate):
    """if len(audio) < 32000:  # pad to 2 seconds
        audio = np.pad(audio, (0, 32000 - len(audio)), "constant")
    elif len(audio) > 32000:
        audio = audio[:32000]"""
    # print(len(audio))
    # SPEC = standardize(librosa.feature.mfcc(y=audio, sr=sample_rate))

    # audio = rmsNorm(audio, -50)
    audio_cqt_spec = np.abs(
        librosa.cqt(
            y=audio,
            sr=sample_rate,
            fmin=librosa.note_to_hz("C1"),
        )
    )

    audio_cqt_spec = librosa.amplitude_to_db(audio_cqt_spec)
    # audio_cqt_spec = minmax_scale(audio_cqt_spec)  # feature_range=(-1, 1))
    audio_cqt_spec = minmax_scale(audio_cqt_spec, feature_range=(-1, 1))

    S = np.expand_dims(audio_cqt_spec, axis=-1)

    return S, sample_rate


def load(path, seconds_limit=(None, None)):
    audio, sample_rate = librosa.load(path, mono=True, sr=16000)

    if seconds_limit[0] != None and seconds_limit[1] != None:
        audio = audio[
            int(seconds_limit[0] * sample_rate) : int(seconds_limit[1] * sample_rate)
        ]

    return audio, sample_rate


def loadAndPrepare(path, audio_limit_sec=(None, None), MAXTIMESTEPS=None):
    audio, sample_rate = load(path, audio_limit_sec)

    if MAXTIMESTEPS != None:
        if len(audio) < MAXTIMESTEPS * (sample_rate * 2):
            audio = np.pad(
                audio,
                (0, int(MAXTIMESTEPS * (sample_rate * 2) - len(audio))),
                "constant",
            )

    # S, _ = Prepare(audio, sample_rate)

    time_steps = []
    # batchs of 2 seconds each
    for i in range(0, len(audio), 32000):
        S, _ = Prepare(audio[i : i + 32000], sample_rate)
        # print(S.shape)
        time_steps.append(S)

    return np.array(time_steps), sample_rate


def standardize(S):
    # Reshape the data to 2D for standardization
    S_2d = S.reshape(-1, S.shape[-1])

    # Standardize the data
    scaler = StandardScaler()
    S_2d_scaled = scaler.fit_transform(S_2d)

    # Reshape the data back to its original shape
    S_scaled = S_2d_scaled.reshape(S.shape)

    return S_scaled
