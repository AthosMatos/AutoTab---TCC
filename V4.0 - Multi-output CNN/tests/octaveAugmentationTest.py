from utils.paths import RAW_DATASETS
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
from utils.audio.load_prepare_CURRENT import Prepare

path = f"{RAW_DATASETS.AthosSet}\\chords\\AMajor\\E2-A2-E3-A3-C♯4-E4\eletric - Chord - A - 1.wav"

y, sr = librosa.load(path, sr=None)

# Perform pitch shifting
y_transposed = librosa.effects.pitch_shift(
    y=y, sr=sr, n_steps=12
)  # 12 steps correspond to one octave

# Export the transposed audio
""" sf.write("normal-Audio.wav", y, sr)
output_path = "transposed_audio.wav"
sf.write(output_path, y_transposed, sr) """

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

ax[0][0].plot(y)
ax[0][1].plot(y_transposed)

librosa.display.specshow(
    Prepare(y, sr, specType="cqt", expand_dims=False, transpose=True, pad=True),
    sr=sr,
    x_axis="time",
    y_axis="cqt_note",
    ax=ax[1][0],
)
librosa.display.specshow(
    Prepare(
        y_transposed, sr, specType="cqt", expand_dims=False, transpose=True, pad=True
    ),
    sr=sr,
    x_axis="cqt_note",
    y_axis="time",
    ax=ax[1][1],
)


plt.show()
