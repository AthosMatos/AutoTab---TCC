import librosa
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from librosa.display import specshow
from librosa.feature import tempo
from sklearn.preprocessing import StandardScaler

figure, axis = plt.subplots(3, 3)

nftt = 2048
hop_length = nftt // 4


def calc_mfcc(audio: np.ndarray, sr: int):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    # Reshape the data to 2D for standardization
    X_2d = mfccs.reshape(-1, mfccs.shape[-1])

    # Standardize the data
    scaler = StandardScaler()
    X_2d_scaled = scaler.fit_transform(X_2d)

    # Reshape the data back to its original shape
    X_scaled = X_2d_scaled.reshape(mfccs.shape)

    mfccs = librosa.util.normalize(X_scaled, norm=1, fill=True)
    print('mfccs shape: ', mfccs.shape)
    return mfccs


def calc_melSpec(audio: np.ndarray, sr: int, n_mels=int):
    S = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=nftt, hop_length=hop_length, n_mels=n_mels)
    S = librosa.power_to_db(S, ref=np.max)

    # Reshape the data to 2D for standardization
    X_2d = S.reshape(-1, S.shape[-1])

    # Standardize the data
    scaler = StandardScaler()
    X_2d_scaled = scaler.fit_transform(X_2d)

    # Reshape the data back to its original shape
    X_scaled = X_2d_scaled.reshape(S.shape)

    S = librosa.util.normalize(X_scaled, norm=1, fill=True)

    """  # S = librosa.util.normalize(S, norm=1, fill=True)
     """
    # chroma = librosa.util.normalize(chroma, norm=1, fill=True)

    print('mel_spec shape: ', S.shape)
    return S


def calc_audio(audio, sr: int):
    # audio = np.mean(audio.T, axis=0) # this is for stereo
    audio = audio.astype(np.float32)
    audio_len = audio.shape[0] / sr
    audio_time = np.linspace(0., audio_len, audio.shape[0])

    print('audio shape: ', audio.dtype)
    return audio, audio_time


def filt(path, index, stereo=False):
    sr, audio = wavfile.read(path)
    if stereo:
        audio = np.mean(audio.T, axis=0)
    audio, audio_time = calc_audio(audio, sr)

    melSpec = calc_melSpec(audio, sr, 256)
    mfccs = calc_mfcc(audio, sr)
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)

    dtempo = tempo(onset_envelope=onset_env, sr=sr,
                   aggregate=None)
    print('beat_times: ', dtempo.__len__())

    axis[0, index].set_title(path)
    axis[0, index].plot(audio_time, audio)

    axis[1, index].set_title('mfcc')
    img = specshow(mfccs, x_axis='time', ax=axis[1, index])
    axis[2, index].set_title('melSpec')
    img = specshow(melSpec, x_axis='time', y_axis='mel', ax=axis[2, index])
    """ axis[2, index].scatter(
        beat_times, [0] * len(beat_times), color='blue', label='Beats') """
    if (index == 0):
        figure.colorbar(img, ax=axis)


filt('./lead.wav', 0)
filt('./rythm.wav', 1)
filt('./2_notes_dist_playing_sequence.wav', 2, True)

plt.show()
