import librosa
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
from utils.audio.load_prepare import loadAndPrepare, load
import numpy as np

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))


path = "simple g test.wav"
path2 = "audio_.wav"

audio, sr = load(path)

print(np.max(audio), np.min(audio), sr)

audio2, sr2 = load(path2)

print(np.max(audio2), np.min(audio2), sr2)

""" spec, sr = loadAndPrepare(
    path, sample_rate=44100, expand_dims=False, transpose=False, pad=False
)
spec2, sr2 = loadAndPrepare(
    path2, sample_rate=44100, expand_dims=False, transpose=False, pad=False
)

print("y.shape: ", spec.shape)
print("y2.shape: ", spec2.shape)

librosa.display.specshow(
    spec,
    sr=sr,
    x_axis="time",
    ax=ax[0],
)
ax[0].set_title("simple g test.wav")
librosa.display.specshow(
    spec2,
    sr=sr2,
    x_axis="time",
    ax=ax[1],
)
ax[1].set_title("audio_.wav")

plt.show() """
