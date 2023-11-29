from keras.models import load_model
import numpy as np
from Helpers.DataPreparation.audioUtils import loadAndPrepare
import os
from Helpers.consts.labels import labels

model = load_model("model.keras")
# 2.1

spec, _ = loadAndPrepare(
    os.path.dirname(__file__) + "/dataset/training/A2/string-A_fret-0_A2_acoustic.wav"
)

y_pred = model.predict(spec.reshape(1, spec.shape[0], spec.shape[1]))
print(y_pred.shape)
exit()
best_pred = np.argmax(y_pred)
confidence = np.max(y_pred) * 100

print(labels[best_pred])
# 2 decimal places
print(f"confidence: {confidence:.2f}%")
