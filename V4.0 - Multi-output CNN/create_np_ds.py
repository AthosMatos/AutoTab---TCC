from multiprocessing import Manager, Process, cpu_count
import os
from typing import List
from utils.audio.load_prepare import loadAndPrepare
from utils.notes import genNotes
import tensorflow as tf
import numpy as np
from utils.paths import CUSTOM_DATASETS


# 30 frets because the dataset has harmonics that go beyond the 22 frets
GUITAR_NOTES, _, GUITAR_NOTES_INDEXES = genNotes(indexes=True, FRETS=30)


""" 
TRANSPOSING THE SPEC- yield very good results, keep that in mind, good predictions and identification of singular notes on a harmony


"""


def map_to_outs(labelnotes: list, uniqueLabels):
    # print("labelnotes", labelnotes)
    # start a np array filled with 'false'
    new_labels = np.full((len(uniqueLabels)), 0)
    for label in labelnotes:
        index_of_label = uniqueLabels.tolist().index(label)
        new_labels[index_of_label] = 1
        # print("index_of_label", index_of_label)

    return new_labels


def getFilesPATHS(Fpaths: [str], extension=".wav"):
    paths = []
    labels_extend = []
    labels = []
    brk = False
    for path in Fpaths:
        # print(f"Loading {path}")
        for DIRS, _, files in os.walk(path):
            for file in files:
                if file.endswith(extension):
                    LABEL = DIRS.split("\\")[-1]
                    labelnotes = LABEL.split("-")
                    if len(labelnotes) > 6:
                        # print(f"Error: {labelnotes} has more than 6 notes")
                        continue

                    paths.append(os.path.join(DIRS, file))
                    labels_extend.extend(labelnotes)
                    labels.append(labelnotes)

                    """ if len(labels) > 100:
                        brk = True
                        break
            if brk:
                break
        if brk:
            break """

    # shuffle paths and labels
    p = np.random.permutation(len(paths))
    """ paths, labels = np.array(paths)[p], np.array(labels)[p] """
    # labels_extend.append("none")
    uniqueLabels = np.unique(labels_extend)
    uniqueIndexes = np.arange(len(uniqueLabels))
    newLabels = []
    for label in labels:
        newLabels.append(map_to_outs(label, uniqueLabels))
    labels = np.array(newLabels)

    # print("newLabels", labels.shape)
    print("unique labels", uniqueLabels)
    # print("uniqueIndexes", uniqueIndexes)

    # print(updadteLabels.shape)
    return np.array(paths)[p], np.array(newLabels)[p], labels_extend, uniqueLabels


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
        audio, _ = loadAndPrepare(
            path, sample_rate=44100, expand_dims=True, pad=True, transpose=True
        )

        loaded_audios.append(audio)
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

    print(f"Preprocessing {FILES_AMOUNT} files with {CPUS} CPUs")

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
    """  print("Shuffling")
    p = np.random.permutation(FILES_AMOUNT - 1)
    loaded_audios, loaded_labels = loaded_audios[p], loaded_labels[p] """

    # split the data to 0.7, 0.2, 0.1
    train_size = int(0.6 * FILES_AMOUNT)
    val_size = int(0.3 * FILES_AMOUNT)
    # test_size = FILES_AMOUNT - (train_size + val_size)

    x_train, y_train = loaded_audios, loaded_labels
    """ x_val, y_val = (
        loaded_audios[train_size : train_size + val_size],
        loaded_labels[train_size : train_size + val_size],
    )
    x_test, y_test = (
        loaded_audios[train_size + val_size :],
        loaded_labels[train_size + val_size :],
    ) """

    print(f"Train: {len(x_train)} Files")
    """ print(f"Val: {len(x_val)} Files")
    print(f"Test: {len(x_test)} Files") """

    # print the output
    """ while not loaded_audios.empty():
        print(loaded_audios.get()) """

    # Perform actions to save or process the loaded data
    # For example, you can save them to a file, process them further, etc.
    print("Saving files")
    path = "np_ds-transposed-new"
    # path = "np_ds"
    # check if the folder exists
    if not os.path.exists(path):
        os.makedirs(path)

    np.savez_compressed(
        f"{path}/train_ds-6out.npz",
        x=x_train,
        y=y_train,
    )
    """  np.savez_compressed(
        "np_ds/val_ds-6out.npz",
        x=x_val,
        y=y_val,
    )
    np.savez_compressed(
        "np_ds/test_ds-6out.npz",
        x=x_test,
        y=y_test,
    ) """
    np.save(f"{path}/unique_labels-6out.npy", UNIQUE_LABELS)

    print("Files saved")

    return True


if __name__ == "__main__":
    main()
