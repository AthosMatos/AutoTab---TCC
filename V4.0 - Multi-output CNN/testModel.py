from keras.models import load_model
import numpy as np
from utils.audio.load_prepare import loadAndPrepare
from utils.paths import CUSTOM_DATASETS, RAW_DATASETS

unique_labels = np.load("all_labels.npy")

# path = RAW_DATASETS.path + "/musics/g chord.wav"
# path = RAW_DATASETS.path + "/musics/beach house - clean.wav"
path = RAW_DATASETS.path + "/musics/riffs test.wav"
"""
    riff test notes in riff
    1- A2 E3 A3
    2- D3 A3 D4
    3 - E3 B3 E4
"""

model = load_model("Models/model_chords.h5")
# model = load_model("Models/model_notes.h5")

spec, _ = loadAndPrepare(
    path,
    sample_rate=44100,
    # audio_limit_sec=(0.38, 0.8),
    audio_limit_sec=(1.23, 2.43),
    # audio_limit_sec=(1.82, 2.12),
    # audio_limit_sec=(2.05, 2.36),
    # audio_limit_sec=(1.5, 1.7),
    notes=False,
)


y_pred = model.predict(spec.reshape(1, spec.shape[0], spec.shape[1], 1))

for i, pred in enumerate(y_pred):
    confidence = np.max(pred) * 100
    if confidence > 50:
        print(f"|| {unique_labels[i]} - {confidence:.2f}% ||")

    """ print(np.argmax(pred))
    print(unique_labels[np.argmax(pred)]) """
