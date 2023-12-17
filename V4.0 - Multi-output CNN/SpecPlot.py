import matplotlib.pyplot as plt
from utils.audio.load_prepare import loadAndPrepare, load, Prepare
import numpy as np
from utils.paths import CUSTOM_DATASETS, RAW_DATASETS

fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))


path = f"{RAW_DATASETS.AthosSet}\\chords\\GMajor\\CHORD - G Major 01.wav"
path2 = f"{RAW_DATASETS.AthosSet}\\chords\\AMajor\\E2-A2-E3-A3-C♯4-E4\eletric - Chord - A - 1.wav"
path3 = f"{RAW_DATASETS.AthosSet}\\chords\\DMajor\\CHORD - D Major 01.wav"
path4 = CUSTOM_DATASETS.IDMT_SMT_GUITAR_V2.notes + "\\A2\\AR_A_fret_1-20_0.wav"
path5 = RAW_DATASETS.AthosSet + "/chords/CMajor/CHORD - C Major 03.wav"
path6 = (
    CUSTOM_DATASETS.IDMT_SMT_GUITAR_V2.chords
    + "\\dataset2\\C#3-F#3-C#4-A3\FS_Lick9_FN_Lage_19.wav"
)
# path2 = CUSTOM_DATASETS.notes + "\\A#4\\00_BN2-166-Ab_solo.wav_1086.wav"
path7 = CUSTOM_DATASETS.GuitarSet.notes + "\\C3\\00_Jazz3-150-C_solo.wav_2146.wav"
# path2 = CUSTOM_DATASETS.notes + "\\A#4\\00_BN2-166-Ab_solo.wav_1086.wav"
path8 = CUSTOM_DATASETS.IDMT_SMT_GUITAR_V2.notes + "\\D3\\AR_A_fret_1-20_5.wav"

sr = 44100

spec, _ = loadAndPrepare(
    path,
    sample_rate=sr,
    transpose=False,
    expand_dims=False,
    pad=True,
    # audio_limit_sec=(0, 2.5),
)

spec2, _ = loadAndPrepare(
    path2,
    sample_rate=sr,
    transpose=False,
    expand_dims=False,
    pad=True,
    # audio_limit_sec=(0, 2.5),
)

spec3, _ = loadAndPrepare(
    path3,
    sample_rate=sr,
    transpose=False,
    expand_dims=False,
    pad=True,
    # audio_limit_sec=(0, 2.5),
)

spec4, _ = loadAndPrepare(
    path4,
    sample_rate=sr,
    transpose=False,
    expand_dims=False,
    pad=True,
    # audio_limit_sec=(0, 2.5),
)

spec5, _ = loadAndPrepare(
    path5,
    sample_rate=sr,
    transpose=False,
    expand_dims=False,
    pad=True,
    # audio_limit_sec=(0, 2.5),
)

spec6, _ = loadAndPrepare(
    path6,
    sample_rate=sr,
    transpose=False,
    expand_dims=False,
    pad=True,
    # audio_limit_sec=(0, 2.5),
)

spec7, _ = loadAndPrepare(
    path7,
    sample_rate=sr,
    transpose=False,
    expand_dims=False,
    pad=True,
    # audio_limit_sec=(0, 2.5),
)

spec8, _ = loadAndPrepare(
    path8,
    sample_rate=sr,
    transpose=False,
    expand_dims=False,
    pad=True,
    # audio_limit_sec=(0, 2.5),
)


print("y.shape: ", spec.shape)
print("y2.shape: ", spec2.shape)
print("y3.shape: ", spec3.shape)
print("y4.shape: ", spec4.shape)
print("y5.shape: ", spec5.shape)

ax[0][0].set_title("GMajor")
ax[0][0].imshow(spec, aspect="auto", origin="lower")

ax[0][1].set_title("AMajor")
ax[0][1].imshow(spec2, aspect="auto", origin="lower")

ax[0][2].set_title("DMajor")
ax[0][2].imshow(spec3, aspect="auto", origin="lower")

ax[0][3].set_title("A2")
ax[0][3].imshow(spec4, aspect="auto", origin="lower")

ax[1][0].set_title("Cmajor")
ax[1][0].imshow(spec5, aspect="auto", origin="lower")

ax[1][1].set_title("C#3-F#3-C#4-A3")
ax[1][1].imshow(spec6, aspect="auto", origin="lower")

ax[1][2].set_title("C3")
ax[1][2].imshow(spec7, aspect="auto", origin="lower")

ax[1][3].set_title("D3")
ax[1][3].imshow(spec8, aspect="auto", origin="lower")

plt.show()
