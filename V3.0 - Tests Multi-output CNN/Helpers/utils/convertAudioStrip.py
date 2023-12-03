import os
import soundfile

""" 
This code reads audio files from a directory and splits them into 1sec audio segments. 
The segments are then saved to a new directory with a file name that 
includes information about the original file and the segment.
"""

strs = ["E", "A", "D", "G", "B", "ee"]
ds_path = os.path.join(os.getcwd(), "dataset")
# 22 frets
strings = [
    [
        "E2",
        "F2",
        "F#2",
        "G2",
        "G#2",
        "A2",
        "A#2",
        "B2",
        "C3",
        "C#3",
        "D3",
        "D#3",
        "E3",
        "F3",
        "F#3",
        "G3",
        "G#3",
        "A3",
        "A#3",
        "B3",
        "C4",
        "C#4",
    ],
    [
        "A2",
        "A#2",
        "B2",
        "C3",
        "C#3",
        "D3",
        "D#3",
        "E3",
        "F3",
        "F#3",
        "G3",
        "G#3",
        "A3",
        "A#3",
        "B3",
        "C4",
        "C#4",
        "D4",
        "D#4",
        "E4",
        "F4",
        "F#4",
    ],
    [
        "D3",
        "D#3",
        "E3",
        "F3",
        "F#3",
        "G3",
        "G#3",
        "A3",
        "A#3",
        "B3",
        "C4",
        "C#4",
        "D4",
        "D#4",
        "E4",
        "F4",
        "F#4",
        "G4",
        "G#4",
        "A4",
        "A#4",
        "B4",
    ],
    [
        "G3",
        "G#3",
        "A3",
        "A#3",
        "B3",
        "C4",
        "C#4",
        "D4",
        "D#4",
        "E4",
        "F4",
        "F#4",
        "G4",
        "G#4",
        "A4",
        "A#4",
        "B4",
        "C5",
        "C#5",
        "D5",
        "D#5",
        "E5",
    ],
    [
        "B3",
        "C4",
        "C#4",
        "D4",
        "D#4",
        "E4",
        "F4",
        "F#4",
        "G4",
        "G#4",
        "A4",
        "A#4",
        "B4",
        "C5",
        "C#5",
        "D5",
        "D#5",
        "E5",
        "F5",
        "F#5",
        "G5",
        "G#5",
    ],
    [
        "E4",
        "F4",
        "F#4",
        "G4",
        "G#4",
        "A4",
        "A#4",
        "B4",
        "C5",
        "C#5",
        "D5",
        "D#5",
        "E5",
        "F5",
        "F#5",
        "G5",
        "G#5",
        "A5",
        "A#5",
        "B5",
        "C6",
        "C#6",
    ],
]

isDist = False
isAcoustic = True

for j in range(len(strs)):
    filename = strs[j]
    wav_file_path = f"{ds_path}/strips/{filename}String Acoustic STRIP.wav"
    data, samplerate = soundfile.read(wav_file_path)
    index = 0
    for i in range(0, len(data), samplerate):
        # check if the folder exists
        if not os.path.exists(f"{ds_path}/strips/{strings[j][index]}"):
            os.mkdir(f"{ds_path}/strips/{strings[j][index]}")

        add = ""
        if isDist:
            add += "_dist"
        if isAcoustic:
            add += "_acoustic"
        soundfile.write(
            f"{ds_path}/strips/{strings[j][index]}/string-{strs[j]}_fret-{index}_{strings[j][index]}{add}.wav",
            data[i : i + samplerate],
            samplerate,
            subtype="PCM_16",
        )
        print("Converted: " + wav_file_path)
        index += 1
