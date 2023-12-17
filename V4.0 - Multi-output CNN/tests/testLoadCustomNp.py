import numpy as np


train_ds, unique_labels = (
    np.load("np_ds/train_ds-6out.npz", allow_pickle=True),
    np.load("np_ds/unique_labels-6out.npy"),
)
print(train_ds["x"][0].shape)
print(train_ds["y"].shape)
""" INPUT_SHAPE = (train_ds["x"].shape[1], train_ds["x"].shape[2], 1)
NUM_LABELS = len(unique_labels)


TRAIN_Y = [
    train_ds["y"][:, 0, :],
    train_ds["y"][:, 1, :],
    train_ds["y"][:, 2, :],
    train_ds["y"][:, 3, :],
    train_ds["y"][:, 4, :],
    train_ds["y"][:, 5, :],
]
 """
