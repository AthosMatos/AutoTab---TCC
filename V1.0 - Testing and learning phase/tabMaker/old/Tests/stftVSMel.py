import librosa
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import os
from librosa.display import specshow
from sklearn.preprocessing import StandardScaler, minmax_scale

figure, axis = plt.subplots(2, 3)

nftt = int(1024 * 4)
hop_length = nftt // 4


def zero1Norm(S):
    X_2d = S.reshape(-1, S.shape[-1])
    # Standardize the data
    scaler = StandardScaler()
    X_2d_scaled = scaler.fit_transform(X_2d)

    # Reshape the data back to its original shape
    S = X_2d_scaled.reshape(X_2d_scaled.shape)

    # Reshape the data to 2D for standardization
    return S


def peakNorm(S):
    max_value = np.max(S)
    S = S / max_value

    return S


def rmsNorm(S, level=-70.0):
    rms_level = level
    # linear rms level and scaling factor
    r = 10**(rms_level / 10.0)
    a = np.sqrt((len(S) * r**2) / np.sum(S**2))

    # normalize
    S = S * a

    return S


def calc_stft(audio: np.ndarray, sr: int, index=int):
    S = librosa.stft(audio, n_fft=nftt, hop_length=hop_length)
    S = np.abs(S)

    if (index == 0):  # 0-1 normalization
        S = zero1Norm(S)
        # verify if type its number

    elif (index == 1):  # peak normalization
        S = peakNorm(S)

    elif (index == 2):  # RMS normalization
        S = rmsNorm(S, level=-20.0)

    S = librosa.amplitude_to_db(S, ref=np.max)
    S[S < int(-56/2)] = np.min(S)
    S = minmax_scale(S, axis=0)
    return S


def calc_melSpec(audio: np.ndarray, sr: int, n_mels=int, index=int):
    S = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=nftt, hop_length=hop_length, n_mels=n_mels)

    if (index == 0):  # 0-1 normalization
        S = zero1Norm(S)

    elif (index == 1):  # peak normalization
        S = peakNorm(S)

    elif (index == 2):  # RMS normalization
        S = rmsNorm(S)

    # S = minmax_scale(S, feature_range=(0, 1), axis=0)
    # S = minmax_scale(S, feature_range=(-1, 1), axis=0)
    S = librosa.power_to_db(S, ref=np.max)
    return S


def calc_audio(audio, sr: int):
    # audio = np.mean(audio.T, axis=0) # this is for stereo

    audio_len = audio.shape[0] / sr
    audio_time = np.linspace(0., audio_len, audio.shape[0])
    return audio, audio_time


def filt(path, index, stereo=False):
    sr, audio = wavfile.read(path)
    if stereo:
        audio = np.mean(audio.T, axis=0)
    audio = audio.astype(np.float16)

    if (index == 0):
        axis[0, index].set_title('0-1 normalization')
        print('0-1 normalization')
    elif (index == 1):
        axis[0, index].set_title('peak normalization')
        print('peak normalization')
    elif (index == 2):
        axis[0, index].set_title('RMS normalization')
        print('RMS normalization')

    # melSpec = calc_melSpec(audio, sr, 128, index=index)
    # 0.12 seconds of data
    stftspec = calc_stft(audio, sr, index=index)

    # print('shapes ', melSpec.shape, stftspec.shape)

    # print('min: ', np.min(melSpec), 'max: ', np.max(melSpec))
    print('min: ', np.min(stftspec), 'max: ', np.max(stftspec))
    axis[1, index].set_title('Mel')
    """ specshow(melSpec, x_axis='time', y_axis='mel',
             sr=sr, ax=axis[0, index], n_fft=nftt, hop_length=hop_length)
    """
    axis[1, index].set_title('Stft')
    img = specshow(stftspec, x_axis='time', y_axis='log',
                   sr=sr, ax=axis[1, index], n_fft=nftt, hop_length=hop_length)


relative_path = os.path.dirname(__file__)


""" filt(relative_path + '/audio/my bron-yr-aur.wav', 0)
filt(relative_path + '/audio/my bron-yr-aur.wav', 1) """
filt(relative_path + '/audio/my bron-yr-aur.wav', 2)
""" plt.title('myBronYaur') """
plt.show()
