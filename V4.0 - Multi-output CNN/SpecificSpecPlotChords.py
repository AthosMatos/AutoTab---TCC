import matplotlib.pyplot as plt
from utils.audio.load_prepare import loadAndPrepare, load, Prepare
import numpy as np
from utils.paths import CUSTOM_DATASETS, RAW_DATASETS
from utils.files.loadFiles import getFilesPATHS, findFilePath

sr = 44100

specs = []

spec1, _ = loadAndPrepare(
    RAW_DATASETS.path + "/musics/riffs test.wav",
    sample_rate=44100,
    notes=False,
    audio_limit_sec=(1.23, 1.43),
)
spec2, _ = loadAndPrepare(
    RAW_DATASETS.path + "/musics/riffs test.wav",
    sample_rate=44100,
    notes=False,
    audio_limit_sec=(1.23, 1.63),
)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 8))

ax[0].imshow(spec1, aspect="auto", origin="lower")
ax[0].set_title("riff Test ")

ax[1].imshow(spec2, aspect="auto", origin="lower")
ax[1].set_title("riff Test + 1sec")


plt.show()
