import xmltodict
from utils.paths import RAW_DATASETS, CUSTOM_DATASETS
from utils.files.loadFiles import getFilesPATHS, findFilePath
import librosa
import soundfile as sf
import os

__DS_NAME__ = "IDMT-SMT-GUITAR_V2"

XMLFILES = getFilesPATHS(RAW_DATASETS.IDMT_SMT_GUITAR_V2, extension=".xml")

NOTE_SAVE_PATH = CUSTOM_DATASETS.IDMT_SMT_GUITAR_V2.notes
if not os.path.exists(NOTE_SAVE_PATH):
    os.makedirs(NOTE_SAVE_PATH)

CHORD_SAVE_PATH = CUSTOM_DATASETS.IDMT_SMT_GUITAR_V2.chords
if not os.path.exists(CHORD_SAVE_PATH):
    os.makedirs(CHORD_SAVE_PATH)


def save_note():
    if curr_end_seconds - curr_start_seconds < 0.25:
        # print("Note audio file too short")
        return
    SAVEPATH = f"{NOTE_SAVE_PATH}\\{curr_note}".replace("♯", "#")
    if not os.path.exists(SAVEPATH):
        os.makedirs(SAVEPATH)
    E_NOTE_FILENAME = f"{FILE_NAME_CLEAN}_{i}.wav"
    # check if there is a file with the same name
    if os.path.exists(f"{SAVEPATH}\\{E_NOTE_FILENAME}"):
        print(f"File {SAVEPATH}\\{E_NOTE_FILENAME} already exists")
        return
    sf.write(
        f"{SAVEPATH}\\{E_NOTE_FILENAME}",
        FILEWAV[int(curr_start_samples) : int(curr_end_samples)],
        SR,
        "PCM_16",
    )
    print(curr_note)


def save_chord():
    # get the first E_START_SAMPLES and the last E_END_SAMPLES
    E_START_SAMPLES = notes_for_chord[0][3]
    E_END_SAMPLES = notes_for_chord[-1][4]
    E_START_SECONDS = notes_for_chord[0][1]
    E_END_SECONDS = notes_for_chord[-1][2]

    if E_END_SECONDS - E_START_SECONDS < 0.9:
        # print("Note audio file too short")
        return

    notes = ""
    for note in notes_for_chord:
        if len(notes) == 0:
            notes = f"{note[0]}"
        else:
            notes += f"-{note[0]}"

    SAVEPATH = f"{CHORD_SAVE_PATH}\\{SUBFOLDERS}{notes}".replace("♯", "#")
    if not os.path.exists(SAVEPATH):
        os.makedirs(SAVEPATH)
    file_name = f"{FILE_NAME_CLEAN}_{i}.wav"

    # print("Filename: ", file_name)

    # check if there is a file with the same name
    if os.path.exists(f"{SAVEPATH}\\{file_name}"):
        print(f"File {SAVEPATH}\\{file_name} already exists")
        return

    sf.write(
        f"{SAVEPATH}\\{file_name}",
        FILEWAV[int(E_START_SAMPLES) : int(E_END_SAMPLES)],
        SR,
        "PCM_16",
    )
    print(notes)


def get_seconds_samples_note(EVENT):
    E_NOTE = librosa.midi_to_note(float(EVENT["pitch"]))

    E_START_SECONDS = float(EVENT["onsetSec"])
    E_END_SECONDS = float(EVENT["offsetSec"])

    E_START_SAMPLES = (float(EVENT["onsetSec"])) * SR
    E_END_SAMPLES = (float(EVENT["offsetSec"])) * SR

    return (
        E_NOTE,
        E_START_SECONDS,
        E_END_SECONDS,
        E_START_SAMPLES,
        E_END_SAMPLES,
    )


def prepareXML(xml_string):
    AUDIO_DATA = xmltodict.parse(xml_string)["instrumentRecording"]
    FILE_NAME = AUDIO_DATA["globalParameter"]["audioFileName"]
    FILE_NAME_CLEAN = FILE_NAME.split("\\")[-1].split(".")[0]
    print(FILE_NAME_CLEAN)

    AUDIO_EVENTS = AUDIO_DATA["transcription"]["event"]
    # if is a list
    if type(AUDIO_EVENTS) != list:
        AUDIO_EVENTS = [AUDIO_EVENTS]

    return AUDIO_EVENTS, FILE_NAME_CLEAN


def prepareSubFoldersString():
    # split the wav file path from the DS_NAME foward
    subF = WAVFILEPATH.split(__DS_NAME__ + "\\")[-1].split("\\")
    filename = subF[-1]
    subF = subF[:-2]
    sub = ""
    for i, s in enumerate(subF):
        if i == 0:
            sub = s + "\\"
        else:
            sub += s + "\\"

    # print("filename: ", filename)
    print("sub: ", sub)
    return sub


# for now ignoring dataset4

for XMLFILE in XMLFILES:
    # FILEXML = findFilePath("LP_Lick6_FN.xml", OLD_DS_PATH)
    print("FILEXML: ", XMLFILE)
    XMLPATH = XMLFILE.split("\\annotation\\")[0]
    print("XMLPATH: ", XMLPATH)
    WAVFILENAME = XMLFILE.split("\\")[-1].split(".")[0] + ".wav"
    print("WAVFILENAME: ", WAVFILENAME)
    WAVFILEPATH = findFilePath(WAVFILENAME, XMLPATH)

    if WAVFILEPATH is None:
        # print(f"File {WAVFILENAME} not found")
        continue
    else:
        print(f"File {WAVFILENAME} found")
    FILEWAV, SR = librosa.load(WAVFILEPATH)
    SUBFOLDERS = prepareSubFoldersString()

    with open(XMLFILE, "r") as f:
        xml_string = f.read()
        AUDIO_EVENTS, FILE_NAME_CLEAN = prepareXML(xml_string)

        notes_for_chord = []
        for i in range(0, len(AUDIO_EVENTS)):
            (
                curr_note,
                curr_start_seconds,
                curr_end_seconds,
                curr_start_samples,
                curr_end_samples,
            ) = get_seconds_samples_note(AUDIO_EVENTS[i])

            def save(notes_for_chord):
                if len(notes_for_chord) == 0:
                    save_note()
                    return notes_for_chord
                else:
                    save_chord()
                    return []

            if i >= len(AUDIO_EVENTS) - 1:
                notes_for_chord = save(notes_for_chord)
                break
            (
                next_note,
                next_start_seconds,
                next_end_seconds,
                next_start_samples,
                next_end_samples,
            ) = get_seconds_samples_note(AUDIO_EVENTS[i + 1])

            # check if nextNote has come before the previous note has ended
            if next_start_seconds < curr_end_seconds:
                curr_end_seconds = next_start_seconds
                curr_end_samples = next_start_samples

            if next_start_seconds - curr_start_seconds > 0.03:
                notes_for_chord = save(notes_for_chord)

            else:
                if len(notes_for_chord) == 0:
                    notes_for_chord.extend(
                        [
                            [
                                curr_note,
                                curr_start_seconds,
                                curr_end_seconds,
                                curr_start_samples,
                                curr_end_samples,
                            ],
                            [
                                next_note,
                                next_start_seconds,
                                next_end_seconds,
                                next_start_samples,
                                next_end_samples,
                            ],
                        ]
                    )
                else:
                    notes_for_chord.append(
                        [
                            next_note,
                            next_start_seconds,
                            next_end_seconds,
                            next_start_samples,
                            next_end_samples,
                        ]
                    )
