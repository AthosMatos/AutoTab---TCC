import os
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
from audioUtils import loadAndPrepare


def standardize(S):
    # Reshape the data to 2D for standardization
    S_2d = S.reshape(-1, S.shape[-1])

    # Standardize the data
    scaler = StandardScaler()
    S_2d_scaled = scaler.fit_transform(S_2d)

    # Reshape the data back to its original shape
    S_scaled = S_2d_scaled.reshape(S.shape)

    return S_scaled


def gen_X():
    x = []

    index = 0
    path = os.path.dirname(__file__) + "/newDS/beachHouse/"

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".wav"):
                """if index == TIME_STEPS:
                x.append(xx)
                index = 0"""

                xx = []
                filepath = os.path.join(root, file)
                spec, _ = loadAndPrepare(filepath)

                x.append(spec)
                # print(filepath)

                """ index += 1 """

    x = np.array(x)
    X = x.reshape(1, x.shape[0], x.shape[1], x.shape[2], 1)

    print(f"Xtrain shape {X.shape}")

    return X, x
