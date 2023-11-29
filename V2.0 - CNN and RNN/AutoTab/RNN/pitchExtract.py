import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import numpy as np


def rmsNorm(S, level=-70.0):
    rms_level = level
    # linear rms level and scaling factor
    r = 10 ** (rms_level / 10.0)
    a = np.sqrt((len(S) * r**2) / np.sum(S**2))

    # normalize
    S = S * a

    return S


# Load an audio file
audio_file = os.path.dirname(__file__) + "/newDS/my bron-yr-aur.mp3"
y, sr = librosa.load(audio_file, sr=16000, duration=5)

# Load an audio file
audio_file2 = os.path.dirname(__file__) + "/newDS/string-E_fret-0_C2_acoustic.wav"
y2, sr2 = librosa.load(audio_file2, sr=16000, duration=5)

# limit the stft to up to 2000hz
stft = np.abs(librosa.stft(y=y, win_length=1024, hop_length=1024))
stft_limited = stft[:256, :]
stft_limited = rmsNorm(stft_limited)
print(stft_limited.shape)

stft2 = np.abs(librosa.stft(y=y2, win_length=1024, hop_length=1024))
stft_limited2 = stft2[:256, :]
stft_limited2 = rmsNorm(stft_limited2)
print(stft_limited2.shape)

fig, ax = plt.subplots(2, 1)

librosa.display.specshow(stft_limited, sr=sr, x_axis="time", y_axis="hz", ax=ax[0])
librosa.display.specshow(stft_limited2, sr=sr2, x_axis="time", y_axis="hz", ax=ax[1])

plt.show()
