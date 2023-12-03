import librosa
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler, minmax_scale


def rmsNorm(S, level=-70.0):
    rms_level = level
    # linear rms level and scaling factor
    r = 10**(rms_level / 10.0)
    a = np.sqrt((len(S) * r**2) / np.sum(S**2))

    # normalize
    S = S * a

    return S


relative_path = os.path.dirname(__file__)

sr, audio = wavfile.read(relative_path + '/audio/my bron-yr-aur.wav')
if audio.shape.__len__() > 1:
    audio = np.mean(audio.T, axis=0)
audio = audio.astype(np.float16)

S = librosa.stft(audio, n_fft=int(1024 * 8), hop_length=256)
S = np.abs(S)
S = rmsNorm(S, level=-10.0)
S = librosa.amplitude_to_db(S, ref=np.max)
S = minmax_scale(S, axis=0)
S[S < 0.55] = 0

librosa.display.specshow(
    S, x_axis='time', y_axis='log', sr=sr)

plt.show()
