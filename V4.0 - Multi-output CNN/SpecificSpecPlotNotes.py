import matplotlib.pyplot as plt
from utils.audio.load_prepare_prev import loadAndPrepare, load, Prepare
import numpy as np
from utils.paths import CUSTOM_DATASETS, RAW_DATASETS
from utils.files.loadFiles import getFilesPATHS, findFilePath

sr = 44100

specs = []

spec1, _ = loadAndPrepare(
    CUSTOM_DATASETS.IDMT_SMT_GUITAR_V2.notes + "/G4/AR_D_fret_1-20_17.wav",
    sample_rate=44100,
    notes=True,
)
spec2, _ = loadAndPrepare(
    RAW_DATASETS.path + "/musics/beach house - clean.wav",
    sample_rate=44100,
    audio_limit_sec=(2.05, 2.36),
    notes=True,
)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 8))

ax[0].imshow(spec1, aspect="auto", origin="lower")
ax[0].set_title("G4")
ax[1].imshow(spec2, aspect="auto", origin="lower")
ax[1].set_title("Beach House")


plt.show()
