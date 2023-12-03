import os
from scipy.io import wavfile
import numpy as np
import librosa
from sklearn.preprocessing import minmax_scale


class LoadAudio:
    def __init__(self, fileName):
        relative_path = os.path.dirname(__file__) + "/../audio/"
        self.path = relative_path + fileName
        self.sr, self.audio = wavfile.read(self.path)
        if self.audio.shape.__len__() > 1:
            self.audio = np.mean(self.audio.T, axis=0)
        self.audio = self.audio.astype(np.float16)
        self.samplesAmount = self.audio.shape[0]
        self.nftt = int(1024 * 4)
        self.hop_length = self.nftt // 4

    def getAudio(self):
        return self.audio

    def getStft(self):
        S = librosa.stft(self.audio, n_fft=self.nftt,
                         hop_length=self.hop_length)
        S = np.abs(S)
        S = self.rmsNorm(S, level=-20.0)
        S = librosa.amplitude_to_db(S, ref=np.max)
        # S = minmax_scale(S, axis=0)
        return S

    def getSr(self):
        return self.sr

    def getNftt(self):
        return self.nftt

    def getHopLength(self):
        return self.hop_length

    def getAudioTime(self):
        audio_len = self.audio.shape[0] / self.sr
        audio_time = np.linspace(0., audio_len, self.audio.shape[0])
        return audio_time

    def getAudioLength(self):
        return self.samplesAmount / self.sr

    def getAudioShape(self):
        return self.audio.shape

    def getAudioSampleRate(self):
        return self.sr

    def getSamplesAmount(self):
        return self.samplesAmount

    def rmsNorm(self, S, level=-70.0):
        rms_level = level
        # linear rms level and scaling factor
        r = 10**(rms_level / 10.0)
        a = np.sqrt((len(S) * r**2) / np.sum(S**2))

        # normalize
        S = S * a

        return S
