import librosa
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from librosa.display import specshow
figure, axis = plt.subplots(2)

nftt = 8192
hop_length = nftt // 4


def calc_melSpec(audio: np.ndarray, sr: int):
    S = librosa.feature.melspectrogram(
        audio, sr, n_fft=nftt, hop_length=hop_length)
    # S_dB = librosa.power_to_db(S, ref=np.max)
    # chroma = librosa.util.normalize(chroma, norm=1, fill=True)

    print('mel_spec shape: ', S.shape)
    return S


def calc_audio(audio: np.ndarray, sr: int):
    audio = np.mean(audio.T, axis=0)
    audio_len = audio.shape[0] / sr
    audio_time = np.linspace(0., audio_len, audio.shape[0])

    print('audio shape: ', audio.shape)
    return audio, audio_time


def filt(path):
    sr, audio = wavfile.read(path)
    audio, audio_time = calc_audio(audio, sr)

    melSpec = calc_melSpec(audio, sr)

    axis[0].set_title(path)
    axis[0].plot(audio_time, audio)

    axis[1].set_title('melSpec')
    img = specshow(melSpec, x_axis='time', y_axis='mel', sr=sr, ax=axis[1])

    figure.colorbar(img, ax=axis)


filt('./octavesOfC.wav')

plt.show()
