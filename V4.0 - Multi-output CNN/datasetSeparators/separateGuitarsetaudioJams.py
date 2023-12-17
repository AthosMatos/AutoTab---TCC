import os
from utils.paths import RAW_DATASETS, CUSTOM_DATASETS
import numpy as np
import jams
import librosa
import soundfile as sf
from utils.files.loadFiles import getFilesPATHS, findFilePath

__DS_NAME__ = "GuitarSet"

FILESJAMS = getFilesPATHS(RAW_DATASETS.GuitarSet, extension=".jams")

if not os.path.exists(CUSTOM_DATASETS.GuitarSet.notes):
    os.makedirs(CUSTOM_DATASETS.GuitarSet.notes)

if not os.path.exists(CUSTOM_DATASETS.GuitarSet.chords):
    os.makedirs(CUSTOM_DATASETS.GuitarSet.chords)


def prepareSubFoldersString():
    # split the wav file path from the DS_NAME foward
    subFolders = []
    for WAVFP in WAVFILEPATH:
        subF = WAVFP.split(__DS_NAME__ + "\\")[-1].split("\\")
        # print("subF: ", subF)
        filename = subF[-1]
        if len(subF) > 2:
            subF = subF[:-2]
        else:
            subF = subF[:-1]
        sub = ""
        for i, s in enumerate(subF):
            if i == 0:
                sub = s + "\\"
            else:
                sub += s + "\\"

        # print("filename: ", filename)
        # print("sub: ", sub)
        subFolders.append(sub)

    return subFolders


def loadJams(jams_file_path):
    # Load the JAMS file
    jam = jams.load(jams_file_path)
    notes = []
    onsets = []

    annotations = jam.search(namespace="note_midi")
    for anotation in annotations:
        for i, anotation.data[i] in enumerate(anotation.data):
            start_time = anotation.data[i].time
            end_time = anotation.data[i].time + anotation.data[i].duration
            pitch = anotation.data[i].value
            note = librosa.midi_to_note(pitch)
            # Append the note to the list of notes
            notes.append(note)
            # Append the start and end times to the list of timesteps
            onsets.append((start_time, end_time))
            # Now you can use the start_time and end_time to extract the audio of the note or chord

    onsets = np.array(onsets)
    notes = np.array(notes)
    return notes, onsets


FileIndex = 0


def addFileIndex():
    global FileIndex
    FileIndex += 1


def prepareJams(jams_file_path):
    jam = jams.load(jams_file_path)
    audio_ev = []

    def fillAudioEvents():
        annotations = jam.search(namespace="note_midi")
        for anotation in annotations:
            for i in range(0, len(anotation.data) - 1):
                curr_start_time = anotation.data[i].time
                curr_end_time = anotation.data[i].time + anotation.data[i].duration
                curr_pitch = anotation.data[i].value
                curr_note = librosa.midi_to_note(curr_pitch)
                audio_ev.append(
                    {
                        "note": curr_note,
                        "start_time": curr_start_time,
                        "end_time": curr_end_time,
                    }
                )

    fillAudioEvents()
    # order it by start_time
    audio_ev = sorted(audio_ev, key=lambda k: k["start_time"])
    """ print("WAVFILENAME: ", WAVFILENAME)
    for i, ev in enumerate(audio_ev):
        print(f"{i} - {ev}") """

    def findHarmonics():
        def save_note(nt, start, end):
            if end - start < 0.2:
                # print("Note audio file too short")
                return

            start = start * SR
            end = end * SR

            SAVEPATH = f"{CUSTOM_DATASETS.GuitarSet.notes}\\{nt}".replace("♯", "#")
            if not os.path.exists(SAVEPATH):
                os.makedirs(SAVEPATH)
            E_NOTE_FILENAME = f"{WAVFILENAME}_{FileIndex}.wav"
            # check if there is a file with the same name
            if os.path.exists(f"{SAVEPATH}\\{E_NOTE_FILENAME}"):
                # print(f"File {SAVEPATH}\\{E_NOTE_FILENAME} already exists")
                return
            sf.write(
                f"{SAVEPATH}\\{E_NOTE_FILENAME}",
                FILEWAV[int(start) : int(end)],
                SR,
                "PCM_16",
            )
            addFileIndex()
            # print(curr_note)

        def save_chord(nts, start, end):
            if end - start < 0.5:
                # print("Chord audio file too short")
                return

            start = start * SR
            end = end * SR

            notes = ""
            for note in nts:
                if len(notes) == 0:
                    notes = f"{note}"
                else:
                    notes += f"-{note}"

            SAVEPATH = f"{CUSTOM_DATASETS.GuitarSet.chords}\\{notes}".replace("♯", "#")
            if not os.path.exists(SAVEPATH):
                os.makedirs(SAVEPATH)
            file_name = f"{WAVFILENAME}_{FileIndex}.wav"

            # print("Filename: ", file_name)

            # check if there is a file with the same name
            if os.path.exists(f"{SAVEPATH}\\{file_name}"):
                # print(f"File {SAVEPATH}\\{file_name} already exists")
                return

            sf.write(
                f"{SAVEPATH}\\{file_name}",
                FILEWAV[int(start) : int(end)],
                SR,
                "PCM_16",
            )
            addFileIndex()
            # print(notes)

        harmonics = []
        a_ev = []
        for i in range(0, len(audio_ev)):
            curr_start_time = audio_ev[i]["start_time"]
            curr_end_time = audio_ev[i]["end_time"]
            curr_note = audio_ev[i]["note"]

            if i >= len(audio_ev) - 1:
                save(harmonics, curr_end_time)
                harmonics = []

                break

            next_start_time = audio_ev[i + 1]["start_time"]
            next_end_time = audio_ev[i + 1]["end_time"]
            next_note = audio_ev[i + 1]["note"]

            def save(harmonics, curr_end_time):
                if len(harmonics) == 0:
                    endT = curr_end_time
                    if next_start_time < endT:
                        endT = next_start_time

                    # curr_end_time = curr_end_time - 0.03
                    save_note(
                        curr_note,
                        curr_start_time,
                        endT,
                    )
                    """ a_ev.append(
                        {
                            "note": curr_note,
                            "start_time": curr_start_time,
                            "end_time": curr_end_time,
                        }
                    ) """
                else:
                    start = harmonics[0][1]
                    end = harmonics[-1][2]

                    if next_start_time < end:
                        end = next_start_time

                    # end = end - 0.03
                    nots = []
                    for h in harmonics:
                        nots.append(h[0])

                    save_chord(nots, start, end)
                    """ a_ev.append(
                        {
                            "note": nots,
                            "start_time": start,
                            "end_time": end,
                        }
                    ) """

            if next_start_time - curr_start_time <= 0.03:
                if len(harmonics) == 0:
                    harmonics.extend(
                        [
                            [
                                curr_note,
                                curr_start_time,
                                curr_end_time,
                            ],
                            [
                                next_note,
                                next_start_time,
                                next_end_time,
                            ],
                        ]
                    )
                else:
                    harmonics.append(
                        [
                            next_note,
                            next_start_time,
                            next_end_time,
                        ]
                    )
            else:
                save(harmonics, curr_end_time)
                harmonics = []

        return a_ev

    audio_ev = findHarmonics()
    """ for i, ev in enumerate(audio_ev):
        print(f"{i} - {ev}") """

    return audio_ev


filesRead = 0

for JAMFILE in FILESJAMS:
    # print("JAMFILE: ", JAMFILE)
    WAVFILENAME = JAMFILE.split("\\")[-1].split(".")[0] + ".wav"
    # print("JAMPATH: ", WAVFILENAME)
    # print porcentage of files loaded
    porcentage = (filesRead * 100) / len(FILESJAMS)
    print(f"Loaded {filesRead} of {len(FILESJAMS)} files ({porcentage:.2f}%)")
    WAVFILEPATH = findFilePath(
        WAVFILENAME, RAW_DATASETS.GuitarSet, pathCustomEnd=["_mix", "_mic"]
    )
    WAVFILEPATH_MIC, WAVFILEPATH_MIX = WAVFILEPATH
    if WAVFILEPATH is None:
        # print(f"File {WAVFILENAME} not found")
        continue

    for WAVFP in WAVFILEPATH:
        FILEWAV, SR = librosa.load(WAVFP)
        AUDIO_EVENTS = prepareJams(JAMFILE)

    filesRead = filesRead + 1
