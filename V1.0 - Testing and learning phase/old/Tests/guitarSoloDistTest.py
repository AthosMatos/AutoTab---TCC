import librosa
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from librosa.display import specshow
figure, axis = plt.subplots(3, 3)

nftt = 2048
hop_length = nftt // 4


def calc_melSpec(audio: np.ndarray, sr: int, n_mels=int):
    S = librosa.feature.melspectrogram(
        audio, sr, n_fft=nftt, hop_length=hop_length, n_mels=n_mels)
    S = librosa.power_to_db(S)
    # S = librosa.power_to_db(S)
    """  # S = librosa.util.normalize(S, norm=1, fill=True)
    f0 = librosa.yin(audio,
                     fmin=librosa.note_to_hz('C2'),
                     fmax=librosa.note_to_hz('C7'))
    times = librosa.times_like(f0) """
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

    melSpec2 = calc_melSpec(audio, sr, 256)

    axis[0, index].set_title(path)
    axis[0, index].plot(audio_time, audio)

    axis[1, index].set_title('melSpec')
    """ img = specshow(melSpec, x_axis='time', y_axis='mel',
                   sr=sr, ax=axis[1, index]) """

    axis[2, index].set_title('melSpec2')
    img = specshow(melSpec2, x_axis='time', y_axis='mel',
                   sr=sr, ax=axis[2, index])

    figure.colorbar(img, ax=axis[:, index])


filt('./lead.wav', 0)
filt('./rythm.wav', 1)
filt('./2_notes_dist_playing_sequence.wav', 2, True)

plt.show()
