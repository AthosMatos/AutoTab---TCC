import librosa
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from librosa.display import specshow
import os

figure, axis = plt.subplots(4, 2)


def calc_stft(audio: np.ndarray):
    spec2 = librosa.amplitude_to_db(
        np.abs(librosa.stft(y=audio, n_fft=4096)), ref=np.max)
    # spec2 = librosa.util.normalize(spec2, norm=1, fill=True)

    print('stft shape: ', spec2.shape)
    return spec2


def calc_mfcc(audio: np.ndarray, sr: int):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    # mfccs = librosa.util.normalize(mfccs, norm=1, fill=True)

    print('mfccs shape: ', mfccs.shape)
    return mfccs


def calc_chroma_stft(audio: np.ndarray, sr: int):
    num_chroma = 12 * 8
    chroma = librosa.feature.chroma_stft(
        y=audio, sr=sr, n_fft=4096, n_chroma=num_chroma)
    # chroma = librosa.util.normalize(chroma, norm=1, fill=True)

    print('chroma_stft shape: ', chroma.shape)
    return chroma


def calc_chroma2_stft(audio: np.ndarray, sr: int):

    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=4096)
    # chroma = librosa.util.normalize(chroma, norm=1, fill=True)

    print('chroma_stft shape: ', chroma.shape)
    return chroma


def calc_chroma_cqt(audio: np.ndarray, sr: int):
    num_chroma = 12 * 8
    chroma_cqt = librosa.feature.chroma_cqt(n_octaves=1,
                                            y=audio, sr=sr, n_chroma=num_chroma, bins_per_octave=int(num_chroma))
    # chroma_cqt = librosa.util.normalize(chroma_cqt, norm=1, fill=True)

    print('chroma_cqt shape: ', chroma_cqt.shape)
    return chroma_cqt


def calc_audio(audio: np.ndarray, sr: int):
    audio = np.mean(audio.T, axis=0)
    audio_len = audio.shape[0] / sr
    audio_time = np.linspace(0., audio_len, audio.shape[0])

    print('audio shape: ', audio.shape)
    return audio, audio_time


def filt(path, index):
    sr, audio = wavfile.read(path)

    audio, audio_time = calc_audio(audio, sr)
    spec2 = calc_stft(audio)
    mfccs = calc_mfcc(audio, sr)
    chroma = calc_chroma_stft(audio, sr)
    chroma2 = calc_chroma2_stft(audio, sr)
    # chroma_cqt = calc_chroma_cqt(audio, sr)
    """ # Reshape the data to 2D for standardization
    X_2d = spec.reshape(-1, spec.shape[-1])

    # Standardize the data
    scaler = StandardScaler()
    X_2d_scaled = scaler.fit_transform(X_2d)

    # Reshape the data back to its original shape
    X_scaled = X_2d_scaled.reshape(spec.shape) """

    axis[0, index].set_title(path)
    axis[0, index].plot(audio_time, audio)
    axis[1, index].pcolormesh(spec2, vmin=np.min(spec2), vmax=np.max(spec2))
    axis[2, index].set_title('chroma_stft')
    img = specshow(chroma, y_axis='chroma', x_axis='time', ax=axis[2, index])
    axis[3, index].set_title('chroma_stft2')
    img = specshow(chroma2, y_axis='chroma', x_axis='time', ax=axis[3, index])
    figure.colorbar(img, ax=axis[:, index])


relative_path = os.path.dirname(__file__)
filt(relative_path + '/audio/notes_dist_playing.wav', 0)
filt(relative_path + '/audio/2_notes_dist_playing_sequence.wav', 1)

plt.show()
