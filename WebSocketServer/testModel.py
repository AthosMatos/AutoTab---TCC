from keras.models import load_model
import numpy as np
from utils.audio.load_prepare import loadAndPrepare, load

audio, sr = load("test.wav")
print(audio.shape, sr)
exit()

unique_labels = np.load("np_ds-transposed-new/unique_labels-6out.npy")


# path = RAW_DATASETS.path + "/musics/g chord.wav"
# path = RAW_DATASETS.path + "/musics/beach house - clean.wav"
path = "output.wav"

model = load_model("Models/model-out-6.h5")
# model = load_model("Models/model_notes.h5")

spec, _ = loadAndPrepare(
    path, sample_rate=44100, transpose=True, audio_limit_sec=(0.72, 1.02)
)


y_pred = model.predict(spec.reshape(1, spec.shape[0], spec.shape[1], 1))

for i, pred in enumerate(y_pred):
    confidence = np.max(pred) * 100
    # if confidence > 50:
    print(f"|| {unique_labels[i]} - {confidence:.2f}% ||")

    """ print(np.argmax(pred))
    print(unique_labels[np.argmax(pred)]) """
