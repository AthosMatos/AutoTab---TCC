from multiprocessing import Manager, Process, cpu_count
import os
from typing import List
from utils.audio.load_prepare import loadAndPrepare
from utils.notes import genNotes, genNotes_v2
import tensorflow as tf
import numpy as np
from utils.paths import CUSTOM_DATASETS

# 30 frets because the dataset has harmonics that go beyond the 22 frets
GUITAR_NOTES = genNotes_v2("C2", "A6")


np.save(f"all_labels.npy", GUITAR_NOTES)


outPath = "chords_np_cqt_44.1k"
sampleRate = 44100
ignore = ["notes", "Augmented"]


def printConfig():
    os.system("cls")
    print("|| Current config ||")
    print("Output path:", outPath)
    print("Sample rate:", sampleRate)
    # if len(ignore) > 0: # print only if there are ignored folders, case not print none
    if len(ignore) > 0:
        print("Ignored folders:", ignore)
    else:
        print("Ignored folders: none")

    print()


def manualConfiguration():
    global outPath, sampleRate, ignore

    printConfig()

    if input("Notes or chords? ").lower() == "notes":
        outPath += "notes"
    else:
        outPath += "chords"

    printConfig()

    if input("1- 44.1k or 2- 16k? ").lower() == "2":
        sampleRate = 16000
        outPath += "_16k"
    else:
        sampleRate = 44100
        outPath += "_44.1k"

    printConfig()

    while True:
        print("1 - dataset")
        print("2 - std folder")
        print("3 - done")
        folderToIgn = input("Choose type of folders to ignore: ")
        printConfig()

        if folderToIgn == "1":
            print("For multiple datasets, separate them with a comma")

            print("1 - GuitarSet")
            print("2 - IDMT-SMT-GUITAR-V2")
            print("3 - AthosSet")
            print("4 - none")

            ign = input("Choose the dataset to ignore: ").split(",")
            for i in ign:
                if i == "1":
                    ignore.append("GuitarSet")
                if i == "2":
                    ignore.append("IDMT-SMT-GUITAR-V2")
                if i == "3":
                    ignore.append("AthosSet")
            printConfig()

        elif folderToIgn == "2":
            print("For multiple folders, separate them with a comma")

            print("1 - chords")
            print("2 - notes")
            print("3 - augmented")
            print("4 - none")

            ign = input("Choose the folder to ignore: ").split(",")

            for i in ign:
                if i == "1":
                    ignore.append("chords")
                if i == "2":
                    ignore.append("notes")
                if i == "3":
                    ignore.append("augmented")

            printConfig()

        elif folderToIgn == "3":
            break


def map_to_outs(labelnotes: list, uniqueLabels: list):
    # print("labelnotes", labelnotes)
    # start a np array filled with 'false'
    new_labels = np.full((len(uniqueLabels)), 0)
    for label in labelnotes:
        index_of_label = uniqueLabels.index(label)
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
                    if any(x in DIRS for x in ignore):
                        continue
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
    # uniqueLabels = np.unique(labels_extend)
    # uniqueLabels = GUITAR_NOTES

    newLabels = []
    for label in labels:
        newLabels.append(map_to_outs(label, GUITAR_NOTES))
    labels = np.array(newLabels)

    # print("newLabels", labels.shape)
    print("unique labels", GUITAR_NOTES)
    # print("uniqueIndexes", uniqueIndexes)

    # print(updadteLabels.shape)
    return np.array(paths)[p], np.array(newLabels)[p], labels_extend, GUITAR_NOTES


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
            path,
            sample_rate=sampleRate,
            expand_dims=True,
            pad=True,
            transpose=False,
        )

        loaded_audios.append(audio)
        loaded_labels.append(LABELS[index])
    porcentage = (len(loaded_audios) * 100) / FILES_AMOUNT
    os.system("cls")
    print(f"{len(loaded_audios)} of {FILES_AMOUNT} files ({porcentage:.2f}%)")

    return True


def main():
    """if input("Manual config? (y/n) ").lower() == "y":
    manualConfiguration()"""

    CPUS = cpu_count()
    PATHS, LABELS, _, _ = getFilesPATHS([CUSTOM_DATASETS.path])
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

    # path = "np_ds"
    # check if the folder exists
    if not os.path.exists(outPath):
        os.makedirs(outPath)

    np.savez_compressed(
        f"{outPath}/train_ds-6out.npz",
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
    # np.save(f"{outPath}/unique_labels-6out.npy", UNIQUE_LABELS)

    print("Files saved")

    return True


if __name__ == "__main__":
    main()
