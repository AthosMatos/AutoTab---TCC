import matplotlib.pyplot as plt
import numpy as np
from utils.paths import RAW_DATASETS
import tensorflow as tf
import librosa
from scipy.ndimage import median_filter
from scipy.signal import spectrogram, stft
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import deconvolve
from utils.audio.rmsNorm import rmsNorm


def reverberation_reduction(signal, impulse_response):
    clean_signal, _ = deconvolve(signal, impulse_response)
    return clean_signal


def preemphasis(signal, coeff=0.97):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def spectral_subtraction(signal, noise_level=0.001):
    _, _, Sxx = spectrogram(signal)
    _, _, Snn = stft(np.random.randn(len(signal)))

    alpha = 2.0  # Adjust as needed
    Sxx_clean = np.maximum(Sxx - alpha * Snn, 0)

    return Sxx_clean


def smooth_signal(signal, sigma=3):
    return gaussian_filter1d(signal, sigma)


def median_filtering(spectrogram, size=3):
    return median_filter(spectrogram, size=size)


def dynamic_range_compression(spectrogram, compression_factor=2.0):
    return np.log(1 + compression_factor * spectrogram)


def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


SR = 44100
# audio, sr = librosa.load(RAW_DATASETS.path + "/musics/riff test 3 notes.wav", sr=sr)
audio, SR = librosa.load(
    RAW_DATASETS.path + "/musics/my bron-yr-aur.mp3", sr=SR, duration=8
)
# audio = preemphasis(audio)
# audio = librosa.effects.harmonic(audio, margin=0.5)

# spec = tf.signal.stft(audio, frame_length=255, frame_step=128)

""" spec = tf.math.log(spec) """
# spec = tf.transpose(spec)
""" spec = tf.expand_dims(spec, axis=-1)
spec = tf.image.resize(spec, (128, 128)) """


specs = []
specs.append(
    (
        librosa.amplitude_to_db(
            tf.abs(librosa.cqt(audio, sr=SR, fmin=librosa.note_to_hz("C2"))), ref=np.max
        ),
        "cqt",
    )
)

spec = tf.abs(
    librosa.cqt(
        audio,
        sr=SR,
        fmin=librosa.note_to_hz("C2"),
    )
)
spec = rmsNorm(spec, -20)
spec = librosa.amplitude_to_db(spec, ref=np.max)
specs.append(
    (
        spec,
        "my bron-yr-aur",
    )
)
fig, ax = plt.subplots(nrows=specs.__len__(), ncols=1, figsize=(10, 8))

for i in range(specs.__len__()):
    spect = specs[i][0]
    ax[i].set_title(specs[i][1])
    ax[i].imshow(spect, aspect="auto", origin="lower")


""" 
audio = tf.expand_dims(audio, axis=-1)
print(audio.shape)
audio = tf.image.resize(audio, (128, 128))
print(audio.shape) """


""" 

ax[0].imshow(spec, aspect="auto", origin="lower")
ax[0].set_title("riff Test ") """


plt.show()
