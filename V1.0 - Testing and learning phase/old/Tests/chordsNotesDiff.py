import librosa
from scipy.io import wavfile
from python_speech_features import mfcc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

figure, axis = plt.subplots(3, 2)


def filt(path, index):
    sr, audio = wavfile.read(path)
    audio = np.mean(audio.T, axis=0)

    audio_len = audio.shape[0] / sr
    audio_time = np.linspace(0., audio_len, audio.shape[0])
    spec = mfcc(audio, samplerate=sr, nfft=2048, nfilt=20).T
    spec2 = librosa.amplitude_to_db(
        np.abs(librosa.stft(audio, n_fft=2048)), ref=np.max)

    # Reshape the data to 2D for standardization
    X_2d = spec.reshape(-1, spec.shape[-1])

    # Standardize the data
    scaler = StandardScaler()
    X_2d_scaled = scaler.fit_transform(X_2d)

    # Reshape the data back to its original shape
    X_scaled = X_2d_scaled.reshape(spec.shape)

    print(spec.shape)
    print(spec2.shape)
    print(X_scaled.shape)

    axis[0, index].plot(audio_time, audio)
    axis[1, index].imshow(X_scaled, cmap='hot', origin='lower', aspect='auto')
    axis[2, index].pcolormesh(spec2, vmin=np.min(spec2), vmax=np.max(spec2))


filt('./D - FULL.wav', 0)
filt('./D - ded.wav', 1)

plt.show()
