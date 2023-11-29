from keras.models import load_model
import numpy as np
from utils.audio.load_prepare import loadAndPrepare
from utils.paths import CUSTOM_DATASETS, RAW_DATASETS

""" model = load_model("Models/model-out-6-Adam.h5")  # more precise """
model = load_model("Models/model-out-6-Adam-new.h5")
# model = load_model("Models/model-out-6-RMSPROP.h5") #more imaginative

unique_labels = np.load("np_ds-transposed-new/unique_labels-6out.npy")

# path = f"{RAW_DATASETS.AthosSet}\\chords\\GMajor\\CHORD - G Major 01.wav"
# path = f"{RAW_DATASETS.AthosSet}\\chords\\AMajor\\E2-A2-E3-A3-C♯4-E4\eletric - Chord - A - 2.wav"
path = f"{RAW_DATASETS.AthosSet}\\chords\\DMajor\\CHORD - D Major 02.wav"

spec, _ = loadAndPrepare(
    path,
    sample_rate=44100,
    transpose=True,
    expand_dims=True,
    pad=True,
)


y_pred = model.predict(spec.reshape(1, spec.shape[0], spec.shape[1], spec.shape[2]))

for i, pred in enumerate(y_pred):
    confidence = np.max(pred) * 100
    if confidence > 30:
        print(f"|| {unique_labels[i]} - {confidence:.2f}% ||")
    """ print(np.argmax(pred))
    print(unique_labels[np.argmax(pred)]) """

exit()

confidences = []
notes_preds = []
for pred in y_pred:
    confidence = np.max(pred[0]) * 100
    confidences.append(confidence)
    notes_preds.append(unique_labels[np.argmax(pred[0])])

for i, note in enumerate(notes_preds):
    print(f"|| {note} - {confidences[i]:.2f}% ||")
