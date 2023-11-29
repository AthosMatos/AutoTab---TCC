import numpy as np
from scipy.io import wavfile
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler, minmax_scale
import matplotlib.pyplot as plt

nftt = int(1024 * 8)
hop_length = 128


def loadAndPrepare(path, audio_limit_sec=(None, None)):
    sample_rate, audio = wavfile.read(path)
    if audio.shape.__len__() > 1:
        audio = np.mean(audio.T, axis=0)
    audio = audio.astype(np.float16)
    if (audio_limit_sec[0] != None and audio_limit_sec[1] != None):
        seconds = audio.shape[0]/sample_rate
        audio_limitStart = int((audio.shape[0] * audio_limit_sec[0])/seconds)
        audio_limitEnd = int((audio.shape[0] * audio_limit_sec[1])/seconds)
        audio = audio[audio_limitStart:audio_limitEnd]

    # S = rmsNorm(S, level=-20.0) #not tested
    """ S = librosa.stft(audio, n_fft=nftt, hop_length=hop_length)
        S = librosa.amplitude_to_db(S, ref=np.max) """

    S = librosa.feature.melspectrogram(
        y=audio, sr=sample_rate, n_mels=128, fmax=8000)

    S = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=128)
    """ S = librosa.power_to_db(librosa.feature.mfcc(
        y=audio, n_mfcc=128), ref=np.max) """

    # S = minmax_scale(S, axis=0) #didn't help
    # S[S < 0.55] = 0 #didn't help
    # S = rmsNorm(S, level=-20.0)
    S = standardize(S)

    # print('S.shape = ' + str(S.shape))
    if (S.shape[1] < 44):
        # print('padding')
        # S is a 2D array, it always return a value of shape(4097,variable) but it has to always have shape(4097, 87) so we pad it
        S = np.pad(S, ((0, 0), (0, 44 - S.shape[1])), 'constant')
    return S, sample_rate


def Prepare(audio, sample_rate):
    S = librosa.feature.melspectrogram(
        y=audio, sr=sample_rate, n_mels=128, fmax=8000)

    S = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=128)

    # S = minmax_scale(S, axis=0) #didn't help
    # S[S < 0.55] = 0 #didn't help
    # S = rmsNorm(S, level=-20.0)
    S = standardize(S)

    # print('S.shape = ' + str(S.shape))
    if (S.shape[1] < 44):
        # print('padding')
        # S is a 2D array, it always return a value of shape(4097,variable) but it has to always have shape(4097, 87) so we pad it
        S = np.pad(S, ((0, 0), (0, 44 - S.shape[1])), 'constant')
    return S


def load(path, audio_limit_sec=(None, None)):
    sample_rate, audio = wavfile.read(path)
    if audio.shape.__len__() > 1:
        audio = np.mean(audio.T, axis=0)
    audio = audio.astype(np.float16)
    if (audio_limit_sec[0] != None and audio_limit_sec[1] != None):
        seconds = audio.shape[0]/sample_rate
        audio_limitStart = int((audio.shape[0] * audio_limit_sec[0])/seconds)
        audio_limitEnd = int((audio.shape[0] * audio_limit_sec[1])/seconds)
        audio = audio[audio_limitStart:audio_limitEnd]

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
