import librosa
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import os
from librosa.display import specshow
from sklearn.preprocessing import StandardScaler, minmax_scale

figure, axis = plt.subplots(2, 3)

nftt = 1024
hop_length = nftt // 4


def calc_melSpec(audio: np.ndarray, sr: int, n_mels=int, index=int):
    S = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=nftt, hop_length=hop_length, n_mels=n_mels)

    if (index == 0):  # 0-1 normalization
        S = librosa.power_to_db(S, ref=np.max)
        # Reshape the data to 2D for standardization
        X_2d = S.reshape(-1, S.shape[-1])

        # Standardize the data
        scaler = StandardScaler()
        X_2d_scaled = scaler.fit_transform(X_2d)

        # Reshape the data back to its original shape
        X_scaled = X_2d_scaled.reshape(X_2d_scaled.shape)

        S = minmax_scale(X_scaled, axis=0)
    elif (index == 1):  # peak normalization
        max_value = np.max(S)
        S = S / max_value
        S = librosa.power_to_db(S, ref=np.max)

    elif (index == 2):  # RMS normalization
        rms_level = -70.0
        # linear rms level and scaling factor
        r = 10**(rms_level / 10.0)
        a = np.sqrt((len(S) * r**2) / np.sum(S**2))

        # normalize
        S = S * a
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

    melSpec = calc_melSpec(audio, sr, 256, index=index)
    # 0.12 seconds of data
    melSpec2 = calc_melSpec(audio[:int(sr * 0.12)], sr, 256, index=index)

    print('min: ', np.min(melSpec), 'max: ', np.max(melSpec))
    print('min: ', np.min(melSpec2), 'max: ', np.max(melSpec2))
    specshow(melSpec, x_axis='time', y_axis='mel',
             sr=sr, ax=axis[0, index])

    axis[1, index].set_title('1seg')
    specshow(melSpec2, x_axis='time', y_axis='mel',
             sr=sr, ax=axis[1, index])


relative_path = os.path.dirname(__file__)


filt(relative_path + '/audio/rythm.wav', 0)
filt(relative_path + '/audio/rythm.wav', 1)
filt(relative_path + '/audio/rythm.wav', 2)
plt.title('rythm distortion')
plt.show()
