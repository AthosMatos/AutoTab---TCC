import numpy as np

train_ds, val_ds, test_ds, unique_labels = (
    np.load("np_ds/train_ds-6out.npz"),
    np.load("np_ds/val_ds-6out.npz"),
    np.load("np_ds/test_ds-6out.npz"),
    np.load("np_ds/unique_labels-6out.npy"),
)
INPUT_SHAPE = (train_ds["x"].shape[1], train_ds["x"].shape[2], 1)
NUM_LABELS = len(unique_labels)


TRAIN_Y = [
    train_ds["y"][:, 0, :],
    train_ds["y"][:, 1, :],
    train_ds["y"][:, 2, :],
    train_ds["y"][:, 3, :],
    train_ds["y"][:, 4, :],
    train_ds["y"][:, 5, :],
]

VAL_Y = [
    val_ds["y"][:, 0, :],
    val_ds["y"][:, 1, :],
    val_ds["y"][:, 2, :],
    val_ds["y"][:, 3, :],
    val_ds["y"][:, 4, :],
    val_ds["y"][:, 5, :],
]

TEST_Y = [
    test_ds["y"][:, 0, :],
    test_ds["y"][:, 1, :],
    test_ds["y"][:, 2, :],
    test_ds["y"][:, 3, :],
    test_ds["y"][:, 4, :],
    test_ds["y"][:, 5, :],
]
