from multiprocessing import Manager, Process, Queue, current_process, cpu_count
import os
from Helpers.consts.paths import training_ds_path
from typing import List
from Helpers.DataPreparation.audioUtils import loadAndPrepare, load, Prepare
import numpy as np
from keras.utils import to_categorical


def getFilesPATHS(path: str, extension=".wav"):
    paths = []
    labels = []
    for DIRS, _, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                paths.append(os.path.join(DIRS, file))
                labels.append(DIRS.split("\\")[-1])
                # PRINT FOLDER
    # labelsasIndexes convert the string labels to indexes
    uniqueLabels = np.unique(labels)
    # print(uniqueLabels)
    labelsasIndexes = np.unique(labels, return_inverse=True)[1]
    return paths, labels, labelsasIndexes, uniqueLabels


def load_from_path(
    PATHS: list,
    LABELS: list,
    UNIQUE_LABELS: np.ndarray,
    loaded_audios: list,
    loaded_labels: list,
    FILES_AMOUNT: int,
):
    # print(f"{ current_process().name} Loaded {spec.shape}")
    #
    for index, path in enumerate(PATHS):
        # spec, sr = loadAndPrepare(path)
        audio, sample_rate = load(path)
        spec = Prepare(audio, sample_rate)
        loaded_audios.append(spec)
        label = to_categorical(LABELS[index], num_classes=len(UNIQUE_LABELS))
        # print(label.shape)
        loaded_labels.append(label)
    porcentage = (len(loaded_audios) * 100) / FILES_AMOUNT
    os.system("cls")
    print(f"{len(loaded_audios)} of {FILES_AMOUNT} files ({porcentage:.2f}%)")

    return True


def main():
    CPUS = cpu_count()
    PATHS, LABELS, LABELSASINDEX, UNIQUE_LABELS = getFilesPATHS(training_ds_path)
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
                LABELSASINDEX[path_start:path_end],
                UNIQUE_LABELS,
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
    test_size = FILES_AMOUNT - (train_size + val_size)

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
        "np_files/train_ds.npz",
        x=x_train,
        y=y_train,
    )
    np.savez_compressed(
        "np_files/val_ds.npz",
        x=x_val,
        y=y_val,
    )
    np.savez_compressed(
        "np_files/test_ds.npz",
        x=x_test,
        y=y_test,
    )
    np.save("np_files/unique_labels.npy", UNIQUE_LABELS)

    print("Files saved")

    return True


if __name__ == "__main__":
    main()
