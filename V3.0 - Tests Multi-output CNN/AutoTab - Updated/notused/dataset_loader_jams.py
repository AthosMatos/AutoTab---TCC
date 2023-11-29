from multiprocessing import Manager, Process, cpu_count, Value
import os
from Helpers.consts.paths import guitarset_ds_path, ds_path
from typing import List
from Helpers.DataPreparation.audioUtils import loadAndPrepare, load, Prepare
import numpy as np
from keras.utils import to_categorical
from notes import genNotes
import jams
import librosa
import soundfile as sf

LABELS = genNotes()[0]


def getFilesPATHS(path: str, extension=".wav"):
    paths = []
    for DIRS, _, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                paths.append(os.path.join(DIRS, file))
    return paths


def loadJams(jams_file_path):
    # Load the JAMS file
    jam = jams.load(jams_file_path)

    # Initialize empty lists for "notes" and "timesteps"
    hot_encoded_notes = []
    notes = []
    onsets = []
    # durations = []
    lbls = [v for v, _ in enumerate(LABELS)]
    hot = to_categorical(lbls, num_classes=len(LABELS))

    # Extract note annotations
    for annotation in jam.annotations:
        if annotation.namespace == "note_midi":
            for note in annotation.data:
                onset_time = note.time
                # check if onset time is a number that is too close to another onset time
                if len(onsets) > 0:
                    for o in onsets:
                        if np.abs(onset_time - o[0]) < 0.1:
                            continue

                duration = note.duration
                midi_pitch = librosa.midi_to_note(note.value)
                notes.append(midi_pitch)
                hot_encoded_notes.append(hot[LABELS.index(midi_pitch)])
                onsets.append((onset_time, onset_time + duration))

    hot_encoded_notes = np.array(hot_encoded_notes)
    onsets = np.array(onsets)
    return hot_encoded_notes, notes, onsets


def findFilePath(filename, path):
    for root, _, files in os.walk(path):
        for file in files:
            filepath = os.path.join(root, file)
            # print(filepath)

            if filepath.split("\\")[-1] == filename:
                return filepath

    return None


def load_from_path(
    PATHS: List[str],
    loaded_audios: list,
    loaded_labels: list,
    FILES_AMOUNT: int,
    files_loaded: Value,
):
    print(f"Processing {len(PATHS)} files")
    guitarSetPath = ds_path + "/GuitarSet"
    # print(guitarSetPath)
    for index, wavfilepath in enumerate(PATHS):
        # spec, sr = loadAndPrepare(path)
        jamFileName = (
            wavfilepath.split("\\")[-1].split(".")[0].strip("_mic").strip("_mix")
        )

        foundJamFile = findFilePath(jamFileName + ".jams", guitarSetPath)
        # print(foundJamFile)
        if foundJamFile is None:
            continue
        hot_encoded_notes, notes, onsets = loadJams(foundJamFile)
        # notes_specs = []
        loaded_labels.extend(hot_encoded_notes)
        # print(notes_labels.shape)

        music, sr = load(wavfilepath)

        for i, (start, end) in enumerate(onsets):
            # srOnset = int(start * sr)
            # 2 decimal places
            # print(f"|| {int(start * sr):.2f}s - {int(end* sr):.2f}s ||")
            sf.write(
                f"{notes[i]}.wav",
                music[int(start * sr) : int(end * sr)],
                sr,
                subtype="PCM_16",
            )
            """ spec = Prepare(music[int(start * sr) : int(end * sr)], sr)
            # print(spec.shape)
            loaded_audios.append(spec)
        files_loaded.value += 1
        porcentage = (files_loaded.value * 100) / FILES_AMOUNT
        # os.system("cls")
        print(f"{files_loaded.value} of {FILES_AMOUNT} files ({porcentage:.2f}%)") """

    return True


def main():
    CPUS = cpu_count()
    PATHS = getFilesPATHS(guitarset_ds_path)
    PATHS = PATHS[:100]

    FILES_AMOUNT = len(PATHS)
    files_loaded = Value("i", 0)
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
                loaded_audios,
                loaded_labels,
                FILES_AMOUNT,
                files_loaded,
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
    SEPARATED_AUDIOS_LEN = loaded_audios.shape[0]
    print("Shuffling")
    p = np.random.permutation(SEPARATED_AUDIOS_LEN)
    loaded_audios, loaded_labels = loaded_audios[p], loaded_labels[p]

    # split the data to 0.7, 0.2, 0.1
    train_size = int(0.7 * SEPARATED_AUDIOS_LEN)
    val_size = int(0.2 * SEPARATED_AUDIOS_LEN)

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
    print(f"{x_train.shape[0]} files for training")
    print(f"{x_val.shape[0]} files for validation")
    print(f"{x_test.shape[0]} files for testing")
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
    np.save("np_files/unique_labels.npy", LABELS)

    print("Files saved")

    return True


if __name__ == "__main__":
    main()
