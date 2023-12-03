import numpy as np
from Helpers.DataPreparation.audioUtils import load, rmsNorm, standardize
import os
import librosa
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale

fig, ax = plt.subplots(ncols=2, nrows=2, sharex=True)
""" bron = (
    os.path.dirname(__file__) + "/dataset/musics/my bron-yr-aur.mp3"
)   """
c1 = (
    os.path.dirname(__file__) + "/dataset/training/C2/string-E_fret-0_C2_dist.wav"
)  # "/dataset/musics/beach house - clean.wav"
c2 = os.path.dirname(__file__) + "/dataset/training/C2/string-E_fret-0_C2_acoustic.wav"
ad, sr = load(c1)
ad2, sr2 = load(c2)
""" MUSICTESTPATH = os.path.dirname(__file__) + "/dataset/musics/TEST SONG.wav"
ad, sr = load(MUSICTESTPATH) """


# amptodb = "pre" or "post" or False but initialy False
def getSpec(
    ad, sr, rmsStrength=-50, amptodb=False, standardize_audio=False, minMaxScale=False
):
    # ad = rmsNorm(ad, rmsStrength)
    # fmin = librosa.note_to_hz("C1")
    audio_cqt_spec = np.abs(
        librosa.cqt(
            y=ad,
            sr=sr,
            # fmin=fmin,
        )
    )

    audio_chroma = librosa.feature.chroma_cqt(
        C=audio_cqt_spec,
        sr=sr,  # fmin=fmin
    )

    print("CQT Shape: ", audio_cqt_spec.shape)

    if amptodb:
        audio_cqt_spec = librosa.amplitude_to_db(audio_cqt_spec)
    if standardize_audio:
        audio_cqt_spec = standardize(audio_cqt_spec)
    if minMaxScale:
        audio_cqt_spec = minmax_scale(audio_cqt_spec, feature_range=(-1, 1))

    return audio_cqt_spec, audio_chroma


def plotSpec(S2, Chroma, sr, axis, label=""):
    ax[0][axis].set_title(label)
    librosa.display.specshow(S2, y_axis="cqt_hz", x_axis="time", sr=sr, ax=ax[0][axis])
    ax[0][axis].label_outer()

    ax[1][axis].set_title(label)
    librosa.display.specshow(
        Chroma, y_axis="chroma", x_axis="time", sr=sr, ax=ax[1][axis]
    )
    ax[1][axis].label_outer()


# pad ad2 to match 2.5 seconds
ad2 = np.pad(ad2, (0, int(2.5 * sr) - ad2.shape[0]), "constant")

S, cr = getSpec(ad, sr, amptodb=True, minMaxScale=True)
S2, cr2 = getSpec(ad2, sr2, amptodb=True, minMaxScale=True)

""" 
print(S.max(), S.min())
print(S2.max(), S2.min()) """

""" # LIMIT = 216
LIMIT = 431
if S2.shape[1] < LIMIT:
    S2 = np.pad(S2, ((0, 0), (0, LIMIT - S2.shape[1])), "constant")
elif S2.shape[1] > LIMIT:
    S2 = S2[:, 0:LIMIT] """

plotSpec(S, cr, sr, 0, "Bron-Yr-Aur")
plotSpec(S2, cr2, sr, 1, "C2 Note")


plt.show()
