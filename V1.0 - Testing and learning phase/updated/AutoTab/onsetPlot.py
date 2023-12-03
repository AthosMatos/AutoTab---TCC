import librosa
import numpy as np
import matplotlib.pyplot as plt
from audioUtils3 import load, Prepare

# 'dataset/NoteChangeTrain/Pluck - fender - clean.wav'
# 'dataset/NoteChangeTrain/Ringing - fender - clean.wav'

audio, sr = load("dataset/musics/my bron-yr-aur.mp3", (0, 3.7))  # Pluck

o_env = librosa.onset.onset_strength(y=audio, sr=sr)
times = librosa.times_like(o_env, sr=sr)
onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)

print(len(onset_frames))

D = np.abs(librosa.stft(audio))
fig, ax = plt.subplots(nrows=2, sharex=True)
librosa.display.specshow(
    librosa.amplitude_to_db(D, ref=np.max), x_axis="time", y_axis="log", ax=ax[0], sr=sr
)
ax[0].set(title="Power spectrogram")
ax[0].label_outer()
ax[1].plot(times, o_env, label="Onset strength")
ax[1].vlines(
    times[onset_frames],
    0,
    o_env.max(),
    color="r",
    alpha=0.9,
    linestyle="--",
    label="Onsets",
)
ax[1].legend()

plt.show()
