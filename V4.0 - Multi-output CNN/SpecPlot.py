import matplotlib.pyplot as plt
from utils.audio.load_prepare import loadAndPrepare, load, Prepare
import numpy as np
from utils.paths import CUSTOM_DATASETS, RAW_DATASETS
from utils.files.loadFiles import getFilesPATHS, findFilePath

paths = getFilesPATHS(
    CUSTOM_DATASETS.path,
    ignores=[],
    extension=".wav",
    randomize=True,
    maxFiles=10,
)
sr = 44100

specs = []

for path in paths:
    spec, _ = loadAndPrepare(path, sample_rate=sr, pad=True)
    print(spec.shape)
    specs.append(spec)


fig, ax = plt.subplots(nrows=specs.__len__() // 2, ncols=2, figsize=(10, 8))

k = 0
i = 0
j = 0
while i < specs.__len__():
    if i == specs.__len__() // 2:
        k += 1
        j = 0
    if k == 2:
        break
    ax[j][k].imshow(specs[i], aspect="auto", origin="lower")
    ax[j][k].set_title(paths[i].split("\\")[-2])
    i += 1
    j += 1

plt.show()


ax[0][0].set_title("GMajor")
ax[0][0].imshow(spec, aspect="auto", origin="lower")


plt.show()
