from multiprocessing import Manager, Process, cpu_count
import os
from Helpers.consts.paths import newDataset
from typing import List
from Helpers.DataPreparation.audioUtils import load, Prepare
import numpy as np
import tensorflow as tf
from Helpers.utils.notes import genNotes

# 30 frets because the dataset has harmonics that go beyond the 22 frets
GUITAR_NOTES, _, GUITAR_NOTES_INDEXES = genNotes(indexes=True, FRETS=30)


def indexLabels(labels: list):
    return np.unique(labels, return_inverse=True)[1]


def getFilesPATHS(path: str, extension=".wav"):
    paths = []
    filename_labels = []
    for DIRS, _, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                paths.append(os.path.join(DIRS, file))
                filename_labels.append(DIRS.split("\\")[-1])
                # PRINT FOLDER
    # labelsasIndexes convert the string labels to indexes
    labels = []
    labels_extend = []
    for LABEL in filename_labels:
        labelnotes = LABEL.split("-")
        labels_extend.extend(labelnotes)
        labels.append(labelnotes)
    # labels_extend.append("none")
    uniqueLabels = np.unique(labels_extend)

    labelsasIndexes = []
    # labelsasIndexes = index of labels based on the unique labels
    for LABEL in labels_extend:
        for index, uniqueLabel in enumerate(uniqueLabels):
            if uniqueLabel == LABEL:
                labelsasIndexes.append(index)
                break
    labelsasIndexes = np.array(labelsasIndexes)

    def fill_W_none(labelnotes):
        if len(labelnotes) < 6:
            for _ in range(6 - len(labelnotes)):
                labelnotes.append(GUITAR_NOTES[-1])
        return labelnotes

    # print(len(labels))

    updadteLabels = []
    for label in labels:  # [note,note,note,note,note,note]
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
        updadteLabels.append(newLnotes)
    # print(labelnotes)
    updadteLabels = np.array(updadteLabels)

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
        spec = Prepare(audio, sample_rate)
        loaded_audios.append(spec)
        loaded_labels.append(LABELS[index])
    porcentage = (len(loaded_audios) * 100) / FILES_AMOUNT
    os.system("cls")
    print(f"{len(loaded_audios)} of {FILES_AMOUNT} files ({porcentage:.2f}%)")

    return True


""" 
for sparse categorical
print(len(LABELS))
    notes = []
    for LABEL in LABELS:
        labelnotes = LABEL.split("-")
        if len(labelnotes) < 6:
            for _ in range(6 - len(labelnotes)):
                labelnotes.append(GUITAR_NOTES[-1])
            for index, gNote in enumerate(GUITAR_NOTES):
                for k, lNote in enumerate(labelnotes):
                    if gNote == lNote:
                        labelnotes[k] = GUITAR_NOTES_INDEXES[index]

        notes.append(labelnotes)
        print(labelnotes)
    notes = np.array(notes)
    # print(notes[0])
    # print(notes)
"""


def main():
    CPUS = cpu_count()
    PATHS, LABELS, LABELS_EXTENDED, UNIQUE_LABELS = getFilesPATHS(newDataset)
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
    p = np.random.permutation(FILES_AMOUNT)
    loaded_audios, loaded_labels = loaded_audios[p], loaded_labels[p]

    # split the data to 0.7, 0.2, 0.1
    train_size = int(0.7 * FILES_AMOUNT)
    val_size = int(0.2 * FILES_AMOUNT)
    # test_size = FILES_AMOUNT - (train_size + val_size)

    x_train, y_train = loaded_audios[:train_size], loaded_labels[:train_size]
    x_val, y_val = (
        loaded_audios[train_size : train_size + val_size],
        loaded_labels[train_size : train_size + val_size],
    )
    x_test, y_test = (
        loaded_audios[train_size + val_size :],
        loaded_labels[train_size + val_size :],
    )

    print(loaded_audios.shape)
    print(loaded_labels.shape)
    # print the output
    """ while not loaded_audios.empty():
        print(loaded_audios.get()) """

    # Perform actions to save or process the loaded data
    # For example, you can save them to a file, process them further, etc.
    print("Saving files")
    # check if the folder exists
    if not os.path.exists("np_files"):
        os.makedirs("np_files")

    np.savez_compressed(
        "np_files/train_ds-6out.npz",
        x=x_train,
        y=y_train,
    )
    np.savez_compressed(
        "np_files/val_ds-6out.npz",
        x=x_val,
        y=y_val,
    )
    np.savez_compressed(
        "np_files/test_ds-6out.npz",
        x=x_test,
        y=y_test,
    )
    np.save("np_files/unique_labels-6out.npy", UNIQUE_LABELS)

    print("Files saved")

    return True


if __name__ == "__main__":
    main()
