from keras.models import load_model
import numpy as np
from Helpers.DataPreparation.audioUtils import loadAndPrepare
import os


model = load_model("model-out6.keras")
# 2.1
unique_labels = np.load("np_files/unique_labels-6out.npy")


""" spec, _ = loadAndPrepare(
    os.path.dirname(__file__)
    + "\\new-dataset\chords\dataset1\Fender Strat Clean Neck SC Chords\A♯2-F3-A♯3-C♯4-F4\\4-A2-Minor 01_4.wav",
    sample_rate=44100,
) """
spec, _ = loadAndPrepare(
    os.path.dirname(__file__)
    + "\\dataset\\training\AthosSet - chords\CMajor\C Major 01 Acoustic.wav",
    sample_rate=44100,
)
# A Major 03 Acoustic
y_pred = model.predict(spec.reshape(1, spec.shape[0], spec.shape[1]))

confidences = []
notes_preds = []
for pred in y_pred:
    confidence = np.max(pred[0]) * 100
    confidences.append(confidence)
    notes_preds.append(unique_labels[np.argmax(pred[0])])

for i, note in enumerate(notes_preds):
    print(f"|| {note} - {confidences[i]:.2f}% ||")
