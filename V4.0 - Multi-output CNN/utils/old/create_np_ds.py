from multiprocessing import Manager, Process, cpu_count
import os
from typing import List
from utils.audio.load_prepare_CURRENT import Prepare, load
from utils.notes import genNotes
import tensorflow as tf
import numpy as np
from utils.paths import CUSTOM_DATASETS

# 30 frets because the dataset has harmonics that go beyond the 22 frets
GUITAR_NOTES, _, GUITAR_NOTES_INDEXES = genNotes(indexes=True, FRETS=30)


def indexLabels(labels: list):
    return np.unique(labels, return_inverse=True)[1]


def getFilesPATHS(Fpaths: [str], extension=".wav"):
    paths = []
    labels_extend = []
    uniqueLabels = np.array([])
    labels = []
    brk = False
    for path in Fpaths:
        # print(f"Loading {path}")
        for DIRS, _, files in os.walk(path):
            for file in files:
                if file.endswith(extension):
                    paths.append(os.path.join(DIRS, file))
                    LABEL = DIRS.split("\\")[-1]
                    labelnotes = LABEL.split("-")
                    labels_extend.extend(labelnotes)
                    labels.append(labelnotes)

                    """ if len(labels) > 1000:
                        brk = True
                        break
            if brk:
                break
        if brk:
            break """

    # check if labelsextended has none
    if "none" not in labels_extend:
        labels_extend.append("none")
    uniqueLabels = np.unique(labels_extend)

    labelsasIndexes = []
    # labelsasIndexes = index of labels based on the unique labels
    for label in labels_extend:
        labelsasIndexes.append(uniqueLabels.tolist().index(label))

    labelsasIndexes = np.array(labelsasIndexes)

    def fill_W_none(labelnotes):
        if len(labelnotes) < 6:
            for _ in range(6 - len(labelnotes)):
                labelnotes.append(GUITAR_NOTES[-1])
        return labelnotes

    # print(len(labels))
    # print("Filling labels")
    updadteLabels = []
    MAXLABELS = len(labels)
    for label in labels:  # [note,note,note,note,note,note]
        # print porcentage
        if len(label) > 6:
            print(f"Error: {label} has more than 6 notes")
            # remove the data in the current index from the paths
            paths.pop(labels.index(label))
            continue
        porcentage = (len(updadteLabels) * 100) / MAXLABELS
        os.system("cls")
        print(f"{len(updadteLabels)} of {MAXLABELS} files ({porcentage:.2f}%)")
        label = fill_W_none(label)
        newLnotes = []
        # for each on of the possible guitar notes in 22 frets
        for lNote in label:
            # print(lNote)
            index = labels_extend.index(lNote)
            # print(index)
            hot = tf.keras.utils.to_categorical(
                labelsasIndexes[index], num_classes=len(uniqueLabels)
            )
            newLnotes.append(hot)
            # newLnotes.append(LABELS_AS_INDEX[index]) sparse categorical
        # for each on of the 6 notes from the output
        newLnotes = np.array(newLnotes)
        print(newLnotes.shape)
        updadteLabels.append(newLnotes)
    # print(labelnotes)
    updadteLabels = np.array(updadteLabels)
    # print(updadteLabels.shape)
    return paths, updadteLabels, labels_extend, uniqueLabels


def load_from_path(
    PATHS: list,
    LABELS: np.ndarray,
    loaded_audios: list,
    loaded_labels: list,
    FILES_AMOUNT: int,
):
    # print(f"{ current_process().name} Loaded {spec.shape}")
    #
    for index, path in enumerate(PATHS):
        # spec, sr = loadAndPrepare(path)
        audio, sample_rate = load(path, sample_rate=44100)
        spec = Prepare(audio, sample_rate, expand_dims=True)
        loaded_audios.append(spec)
        loaded_labels.append(LABELS[index])
    porcentage = (len(loaded_audios) * 100) / FILES_AMOUNT
    os.system("cls")
    print(f"{len(loaded_audios)} of {FILES_AMOUNT} files ({porcentage:.2f}%)")

    return True


def main():
    CPUS = cpu_count()
    PATHS, LABELS, LABELS_EXTENDED, UNIQUE_LABELS = getFilesPATHS(
        [CUSTOM_DATASETS.path]
    )
    FILES_AMOUNT = len(PATHS)

    print(f"Loading {FILES_AMOUNT} files with {CPUS} CPUs")

    MANAGER = Manager()

    processes: List[Process] = []
    loaded_audios = MANAGER.list()
    loaded_labels = MANAGER.list()

    STEP = (FILES_AMOUNT // CPUS) + 1
    path_start = 0
    path_end = STEP
    for _ in range(CPUS):
        p = Process(
            target=load_from_path,
            args=(
                PATHS[path_start:path_end],
                LABELS[path_start:path_end],
                loaded_audios,
                loaded_labels,
                FILES_AMOUNT,
            ),
        )
        processes.append(p)
        p.start()

        path_start = path_end
        path_end += STEP
        if path_end > FILES_AMOUNT:
            path_end = FILES_AMOUNT

    for p in processes:
        p.join()
    loaded_audios = np.array(list(loaded_audios))
    loaded_labels = np.array(list(loaded_labels))
    print("Shuffling")
    p = np.random.permutation(FILES_AMOUNT - 1)
    loaded_audios, loaded_labels = loaded_audios[p], loaded_labels[p]

    # split the data to 0.7, 0.2, 0.1
    train_size = int(0.7 * FILES_AMOUNT)
    # val_size = int(0.2 * FILES_AMOUNT)
    # test_size = FILES_AMOUNT - (train_size + val_size)

    """ x_train, y_train = loaded_audios[:train_size], loaded_labels[:train_size]
    x_val, y_val = (
        loaded_audios[train_size : train_size + val_size],
        loaded_labels[train_size : train_size + val_size],
    )
    x_test, y_test = (
        loaded_audios[train_size + val_size :],
        loaded_labels[train_size + val_size :],
    ) """
    x_train, y_train = loaded_audios, loaded_labels

    print(f"Train: {len(x_train)} Files")
    """ print(f"Val: {len(x_val)} Files")
    print(f"Test: {len(x_test)} Files") """

    # print the output
    """ while not loaded_audios.empty():
        print(loaded_audios.get()) """

    # Perform actions to save or process the loaded data
    # For example, you can save them to a file, process them further, etc.
    print("Saving files")
    # check if the folder exists
    if not os.path.exists("np_ds"):
        os.makedirs("np_ds")

    np.savez_compressed(
        "np_ds/train_ds-6out.npz",
        x=x_train,
        y=y_train,
    )
    """ np.savez_compressed(
        "np_ds/val_ds-6out.npz",
        x=x_val,
        y=y_val,
    ) """
    """ np.savez_compressed(
        "np_ds/test_ds-6out.npz",
        x=x_test,
        y=y_test,
    ) """
    np.save("np_ds/unique_labels-6out.npy", UNIQUE_LABELS)

    print("Files saved")

    return True


if __name__ == "__main__":
    main()
