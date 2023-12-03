import librosa
import numpy as np
import matplotlib.pyplot as plt
from audioUtils3 import load, Prepare
from librosa.display import waveshow, specshow

# 'dataset/NoteChangeTrain/Pluck - fender - clean.wav'
# 'dataset/NoteChangeTrain/Ringing - fender - clean.wav'

audio, sr = load("dataset/musics/beach house - clean.wav", (0, 0.5))  # Pluck
# plot the audio waveform
D = Prepare(audio, sr)
fig, ax = plt.subplots(nrows=2, sharex=True)

print(D.shape)

specshow(D, x_axis="time", y_axis="log", ax=ax[0], sr=sr)
waveshow(audio, sr=sr, ax=ax[1])

ax[0].set(title="Power spectrogram")
ax[0].label_outer()

ax[1].set(title="Waveform")
ax[1].label_outer()


plt.show()
