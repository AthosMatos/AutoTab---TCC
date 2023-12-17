from utils.paths import CUSTOM_DATASETS, DATASET_UTILS
from utils.files.loadFiles import getFilesPATHS, findFilePath
import soundfile as sf
import os
import librosa
from utils.notes import shiftNote, getHighestLowestNote, genNotes_v2
from multiprocessing import cpu_count, Pool, Manager
import numpy as np

WAVFILES_PATHS = getFilesPATHS(
    CUSTOM_DATASETS.path, ignores=["notes"], extension=".wav"
)
POSSIBLE_NOTES = genNotes_v2("C2", "A6")
MAX_NOTE, MIN_NOTE = getHighestLowestNote(POSSIBLE_NOTES)

shift_limit = 10


def genNewShiftedWavs(
    NOTES: list[str], filename: str, file_label: str, filePath: str, NEWFILEPATH: str
):
    # get the highest and lowest notes in the label
    highestInLabel, lowestInLabel = getHighestLowestNote(NOTES)
    shiftStart = shiftNote(lowestInLabel, -48, noteFloor=MIN_NOTE)
    shiftEnd = shiftNote(highestInLabel, 48, noteCeiling=MAX_NOTE)
    shift = np.random.randint(shiftStart, shiftEnd)
    # print(file_label, shiftStart, shiftEnd)

    shift_count = 0

    """ print()
    print(f"|| from label: {file_label} ||") """

    while shift <= shiftEnd:
        if shift_limit != 0 and shift_count >= shift_limit:
            break
        if shift == 0:
            shift += 1
            shift_count += 1
            continue
        newFileLabel = file_label

        """ shift the notes in the label and replace the old label with the new shifted one """
        for currNoteStr in NOTES:
            shiftedNoteStr, _ = shiftNote(currNoteStr, shift)
            newFileLabel = newFileLabel.replace(currNoteStr, shiftedNoteStr)

        # update the filename
        newFilename = f"{filename.split('.')[0]}_shifted({shift}).wav"

        unshiftedWave, sr = librosa.load(filePath, sr=None, mono=True)
        shiftedWave = librosa.effects.pitch_shift(y=unshiftedWave, sr=sr, n_steps=shift)

        DATASET_UTILS.createPath(os.path.join(NEWFILEPATH, newFileLabel))

        """ print(f"shifted {shift} semitones")
        print(f"to label: {newFileLabel}") """

        sf.write(
            os.path.join(NEWFILEPATH, newFileLabel, newFilename),
            shiftedWave,
            sr,
            "PCM_16",
        )
        shift += 1
        shift_count += 1


def genNewFolderStruct(WAVFILEPATH: str):
    """
    Generate a new folder structure based on the given WAV file path.

    Parameters:
    WAVFILEPATH (str): The path of the WAV file.

    Returns:
    tuple: A tuple containing the following elements:
        - NOTES (list): A list of notes extracted from the file labels.
        - filename (str): The name of the file.
        - fileLabels (str): The labels associated with the file.
        - newSubFolders (str): The new subfolder structure generated based on the WAV file path.

    Example:
    >>> genNewFolderStruct("c:\\Users\\athos\\Desktop\\GitHub\\AutoTab---TCC\\Custom\\audio.wav")
    (['audio'], 'audio.wav', 'Custom', 'c:\\Users\\athos\\Desktop\\GitHub\\AutoTab---TCC\\Custom\\Augmented')
    """
    paths_split = WAVFILEPATH.split("Custom")
    NEW_WAV_PATH = paths_split[0] + "Custom\\" + "Augmented" + paths_split[1]
    newSubFolders = "\\".join(NEW_WAV_PATH.split("\\")[:-2])

    SUB_FOLDERS = NEW_WAV_PATH.split("\\")
    filename = SUB_FOLDERS[-1]
    fileLabels = SUB_FOLDERS[-2]
    NOTES = fileLabels.split("-")

    return NOTES, filename, fileLabels, newSubFolders


def convertFile(path: str, processed_files):
    NOTES, FILENAME, FILELABEL, NEWFILEPATH = genNewFolderStruct(path)
    genNewShiftedWavs(NOTES, FILENAME, FILELABEL, path, NEWFILEPATH)

    # processed_files.get() processed_files.set()
    # print a progress bar at each 10 files processed
    if processed_files.get() % 10 == 0:
        os.system("cls")
        print(f"Processed {processed_files.get()} / {len(WAVFILES_PATHS)} files")
        print(
            f"|{'█' * (processed_files.get() // 10)}{' ' * (100 - (processed_files.get() // 10))}|"
        )
    processed_files.set(processed_files.get() + 1)


if __name__ == "__main__":
    MANAGER = Manager()
    processed_files = MANAGER.Value("i", 0)
    with Pool(cpu_count()) as p:
        p.starmap(
            convertFile,
            [(path, processed_files) for path in WAVFILES_PATHS],
        )
