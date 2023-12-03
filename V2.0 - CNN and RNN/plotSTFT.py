from Helpers.consts.paths import ds_path, training_ds_path
import matplotlib.pyplot as plt
import librosa
from Helpers.DataPreparation.audioUtils import load
import numpy as np
import os

axs = plt.figure(figsize=(16, 8)).subplots(2, 2)


audio, sr = load(ds_path + "/musics/my bron-yr-aur.mp3", (0, 2.5))
audio2, _ = load(training_ds_path + "/C2/string-E_fret-0_C2_acoustic.wav")
# audio2, _ = load(training_ds_path + "/AMajor/A Major 01 Acoustic.wav")

stft = np.abs(librosa.stft(y=audio))
print(stft.shape)
stft2 = np.abs(librosa.stft(y=audio, win_length=1024, hop_length=1024))[:256, :]
print(stft2.shape)

exit()

cqt2 = np.abs(
    librosa.cqt(
        audio2,
        sr=sr,
        fmin=librosa.note_to_hz("C2"),
        n_bins=60 * 2,
        bins_per_octave=12 * 2,
    )
)
print(cqt2.shape)


librosa.display.specshow(
    librosa.amplitude_to_db(cqt, ref=np.max),
    sr=sr,
    x_axis="time",
    y_axis="cqt_hz",
    ax=axs[0][0],
)


librosa.display.specshow(
    librosa.amplitude_to_db(cqt2, ref=np.max),
    sr=sr,
    x_axis="time",
    y_axis="cqt_hz",
    ax=axs[1][0],
)


""" spec = rmsNorm(spec, -40) """
""" print(audio.shape) """


plt.show()
