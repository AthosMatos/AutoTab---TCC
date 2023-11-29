import numpy as np
from Helpers.DataPreparation.audioUtils import load, rmsNorm, standardize
import os
import librosa
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale

fig, ax = plt.subplots(ncols=2, sharex=True)
MUSICTESTPATH = (
    os.path.dirname(__file__) + "/dataset/musics/my bron-yr-aur.mp3"
)  # "/dataset/musics/beach house - clean.wav"
MUSICTESTPATH2 = (
    os.path.dirname(__file__) + "/dataset/training/C2/string-E_fret-0_C2_acoustic.wav"
)
ad, sr = load(MUSICTESTPATH, (0, 5))
ad2, sr2 = load(MUSICTESTPATH2)
""" MUSICTESTPATH = os.path.dirname(__file__) + "/dataset/musics/TEST SONG.wav"
ad, sr = load(MUSICTESTPATH) """


# amptodb = "pre" or "post" or False but initialy False
def getSpec(
    ad, sr, rmsStrength=-50, amptodb=False, standardize_audio=False, minMaxScale=False
):
    ad = rmsNorm(ad, rmsStrength)
    audio_cqt_spec = np.abs(
        librosa.cqt(
            y=ad,
            sr=sr,
            fmin=librosa.note_to_hz("C1"),
        )
    )

    print("CQT Shape: ", audio_cqt_spec.shape)

    if amptodb:
        audio_cqt_spec = librosa.amplitude_to_db(audio_cqt_spec, ref=np.max)
    if standardize_audio:
        audio_cqt_spec = standardize(audio_cqt_spec)
    if minMaxScale:
        audio_cqt_spec = minmax_scale(audio_cqt_spec)

    return audio_cqt_spec


def plotSpec(S2, sr, axis, label=""):
    ax[axis].set_title(label)

    librosa.display.specshow(S2, y_axis="cqt_hz", x_axis="time", sr=sr, ax=ax[axis])
    ax[axis].label_outer()


S = getSpec(ad, sr, amptodb=True, minMaxScale=True)
S2 = getSpec(ad2, sr2, amptodb=True, minMaxScale=True)

print(S.max(), S.min())
print(S2.max(), S2.min())

plotSpec(S, sr, 0, "Bron-Yr-Aur")
plotSpec(S2, sr, 1, "C2 Note")


plt.show()
