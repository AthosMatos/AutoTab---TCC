import librosa
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
from utils.audio.load_prepare import loadAndPrepare
import numpy as np
from utils.paths import CUSTOM_DATASETS, RAW_DATASETS

fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 10))


path = f"{RAW_DATASETS.AthosSet}\\chords\\AMajor\\E2-A2-E3-A3-C♯4-E4\eletric - Chord - A - 1.wav"
# path2 = CUSTOM_DATASETS.notes + "\\A#4\\00_BN2-166-Ab_solo.wav_1086.wav"
path2 = CUSTOM_DATASETS.IDMT_SMT_GUITAR_V2.notes + "\\A2\\LP_Lick10_FN_0.wav"
path3 = CUSTOM_DATASETS.IDMT_SMT_GUITAR_V2.notes + "\\G#5\\FS_E1_fret_1-20_16.wav"

mfcc, sr = loadAndPrepare(path, sample_rate=44100, expand_dims=False)
mfcc2, sr2 = loadAndPrepare(path2, sample_rate=44100, expand_dims=False)
mfcc3, sr3 = loadAndPrepare(path3, sample_rate=44100, expand_dims=False)

print("y.shape: ", mfcc.shape)
print("y2.shape: ", mfcc2.shape)
print("y3.shape: ", mfcc3.shape)

librosa.display.specshow(mfcc, sr=sr, x_axis="time", ax=ax[0], y_axis="mel")
librosa.display.specshow(mfcc2, sr=sr2, x_axis="time", ax=ax[1], y_axis="mel")
librosa.display.specshow(mfcc3, sr=sr3, x_axis="time", ax=ax[2], y_axis="mel")


plt.show()
