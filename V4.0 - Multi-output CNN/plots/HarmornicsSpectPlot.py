import librosa
import matplotlib.pyplot as plt
from utils.audio.load_prepare_CURRENT import loadAndPrepare
import numpy as np
from utils.paths import CUSTOM_DATASETS, RAW_DATASETS


fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 6))


path3notes = RAW_DATASETS.path + "/musics/riff test 3 notes.wav"
path2notes = RAW_DATASETS.path + "/musics/riff test 2 notes.wav"
pathriff = RAW_DATASETS.path + "/musics/riffs test.wav"

spec3n, sr = loadAndPrepare(
    path3notes,
    sample_rate=44100,
    expand_dims=False,
    pad=False,
    transpose=True,
    audio_limit_sec=(0, 2.5),
)
spec2n, sr2 = loadAndPrepare(
    path2notes,
    sample_rate=44100,
    expand_dims=False,
    pad=False,
    transpose=True,
    audio_limit_sec=(0, 2.5),
)
specRiff, sr3 = loadAndPrepare(
    pathriff,
    sample_rate=44100,
    expand_dims=False,
    pad=False,
    transpose=True,
    audio_limit_sec=(0, 2.5),
)


print("spec3 min max: ", np.min(spec3n), np.max(spec3n), "shape: ", spec3n.shape)
print("spec2 min max: ", np.min(spec2n), np.max(spec2n), "shape: ", spec2n.shape)
print(
    "specHigh min max: ", np.min(specRiff), np.max(specRiff), "shape: ", specRiff.shape
)


librosa.display.specshow(spec3n, sr=sr, x_axis="cqt_note", ax=ax[0], y_axis="time")
ax[0].set_title("3 notes")

librosa.display.specshow(spec2n, sr=sr2, x_axis="cqt_note", ax=ax[1], y_axis="time")
ax[1].set_title("2 notes")

librosa.display.specshow(specRiff, sr=sr3, x_axis="cqt_note", ax=ax[2], y_axis="time")
ax[2].set_title("Riff")

plt.show()
