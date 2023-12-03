from keras.models import load_model
from Helpers.DataPreparation.audioUtils import loadAndPrepare
from Helpers.consts.labels import labels
import numpy as np

test_ds = np.load("np_files/test_ds-6out.npz")
y_1 = test_ds["y"][:, 0, :]
y_2 = test_ds["y"][:, 1, :]
y_3 = test_ds["y"][:, 2, :]
y_4 = test_ds["y"][:, 3, :]
y_5 = test_ds["y"][:, 4, :]
y_6 = test_ds["y"][:, 5, :]

model = load_model("model-out6.keras")

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(test_ds["x"], [y_1, y_2, y_3, y_4, y_5, y_6], batch_size=128)
print("test loss, test acc:", results)
