import numpy as np
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler, minmax_scale


def loadAndPrepare(path, audio_limit_sec=(None, None)):
    audio, sample_rate = librosa.load(path, mono=True, sr=None)

    if audio_limit_sec[0] != None and audio_limit_sec[1] != None:
        audio = audio[
            int(audio_limit_sec[0] * sample_rate) : int(
                audio_limit_sec[1] * sample_rate
            )
        ]

    # plot the audio waveform
    D = np.abs(librosa.stft(audio, n_fft=1024, hop_length=256))
    S = librosa.amplitude_to_db(D, ref=np.max)

    S = standardize(S)
    # S = minmax_scale(S, axis=0)  # didn't help
    if S.shape[1] < 87:
        S = np.pad(S, ((0, 0), (0, 87 - S.shape[1])), "constant")
    return S, sample_rate


def Prepare(audio, sample_rate):
    # plot the audio waveform
    D = np.abs(librosa.stft(audio, n_fft=1024, hop_length=256))
    S = librosa.amplitude_to_db(D, ref=np.max)

    S = standardize(S)
    # S = minmax_scale(S, axis=0)  # didn't help
    if S.shape[1] < 87:
        S = np.pad(S, ((0, 0), (0, 87 - S.shape[1])), "constant")
    return S


def load(path, audio_limit_sec=(None, None)):
    audio, sample_rate = librosa.load(path, mono=True, sr=None)

    if audio_limit_sec[0] != None and audio_limit_sec[1] != None:
        audio = audio[
            int(audio_limit_sec[0] * sample_rate) : int(
                audio_limit_sec[1] * sample_rate
            )
        ]

    return audio, sample_rate


def standardize(S):
    # Reshape the data to 2D for standardization
    S_2d = S.reshape(-1, S.shape[-1])

    # Standardize the data
    scaler = StandardScaler()
    S_2d_scaled = scaler.fit_transform(S_2d)

    # Reshape the data back to its original shape
    S_scaled = S_2d_scaled.reshape(S.shape)

    return S_scaled
