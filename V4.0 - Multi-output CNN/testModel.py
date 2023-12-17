from keras.models import load_model
import numpy as np
from utils.audio.load_prepare import loadAndPrepare
from utils.paths import CUSTOM_DATASETS, RAW_DATASETS

""" 

train_ds, unique_labels = (
    np.load("chords_np_cqt_44.1k/train_ds-6out.npz"),
    np.load("all_labels.npy"),
)

print(train_ds["x"].shape)
print(train_ds["y"].shape)
print(unique_labels) 

"""

# path = RAW_DATASETS.path + "/musics/riffs test.wav"
# path = RAW_DATASETS.AthosSet + "/chords/CMajor/CHORD - C Major 03.wav"
model = load_model("Models/model_chords.h5")

# model.summary()

unique_labels = np.load("all_labels.npy")
# unique_labels = np.load("np_ds-cqt-minmax-44100/unique_labels-6out.npy")


""" DMAJOR
|| D3 ||
|| A3 ||
|| D4 ||
|| F#4 ||
"""
path = RAW_DATASETS.path + "/musics/g chord.wav"
spec, _ = loadAndPrepare(
    path,
    sample_rate=44100,
    transpose=False,
    expand_dims=True,
    pad=True,
    # audio_limit_sec=(0.38, 0.8)
    # audio_limit_sec=(1.20, 1.40)
    # audio_limit_sec=(1.80, 2.05)
    # audio_limit_sec=(0, 2.5),
)


y_pred = model.predict(spec.reshape(1, spec.shape[0], spec.shape[1], 1))

# print(y_pred)

for i, pred in enumerate(y_pred):
    confidence = np.max(pred) * 100
    if confidence > 30:
        print(f"|| {unique_labels[i]} - {confidence:.2f}% ||")
    """ print(np.argmax(pred))
    print(unique_labels[np.argmax(pred)]) """
