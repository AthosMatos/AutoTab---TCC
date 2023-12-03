import librosa
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
import os
from Helpers.DataPreparation.audioUtils import load, Prepare
from python_speech_features import mfcc

fig, ax = plt.subplots(ncols=2, nrows=1, sharex=True)
bron = os.path.dirname(__file__) + "/dataset/musics/my bron-yr-aur.mp3"
c1 = (
    os.path.dirname(__file__) + "/dataset/training/notes/C2/string-E_fret-0_C2_dist.wav"
)  # "/dataset/musics/beach house - clean.wav"
# c2 = os.path.dirname(__file__) + "/dataset/training/C2/string-E_fret-0_C2_acoustic.wav"
""" MUSICTESTPATH = os.path.dirname(__file__) + "/dataset/musics/TEST SONG.wav"
"""
bron2 = os.path.dirname(__file__) + "/dataset/musics/TEST SONG.wav"
ad, sr = load(bron, (0, 2.5), sample_rate=16000)
ad2, sr2 = load(c1)
ad = Prepare(ad, sr)
print(sr, ad.shape)

exit()


mfccs = mfcc(
    ad,
    sr,
    nfft=2048,
).T
mfccs2 = mfcc(
    ad2,
    sr2,
    nfft=2048,
).T

print("MFCC Shape: ", mfccs.shape)

ax[0].set_title("MFCC 1")
librosa.display.specshow(mfccs, x_axis="time", sr=sr, ax=ax[0])
ax[0].label_outer()
ax[1].set_title("MFCC 2")
librosa.display.specshow(mfccs2, x_axis="time", sr=sr2, ax=ax[1])
ax[1].label_outer()


plt.show()
