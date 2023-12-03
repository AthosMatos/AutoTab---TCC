from keras.models import load_model
import numpy as np
from Helpers.DataPreparation.audioUtils import Prepare, load
import os
from Helpers.consts.labels import labels

model = load_model("model.keras")
# 2.1

audio, sr = load(os.path.dirname(__file__) + "/dataset/musics/my bron-yr-aur.mp3", (2, 2.2))

spec, _ = Prepare(audio, sr)

y_pred = model.predict(spec.reshape(1, spec.shape[0], spec.shape[1]))
best_pred = np.argmax(y_pred)
confidence = np.max(y_pred) * 100

print(labels[best_pred])
# 2 decimal places
print(f"confidence: {confidence:.2f}%")
