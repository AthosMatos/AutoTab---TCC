from keras.models import load_model
from Helpers.DataPreparation.audioUtils import loadAndPrepare
from Helpers.consts.labels import labels
import numpy as np

test_ds = np.load("np_files/test_ds.npz")

model = load_model("model-[64,64]-tanh-[-1,1].keras")

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(test_ds["x"], test_ds["y"], batch_size=128)
print("test loss, test acc:", results)
