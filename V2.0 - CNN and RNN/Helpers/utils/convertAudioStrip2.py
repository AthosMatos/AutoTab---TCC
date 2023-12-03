import os
import soundfile
from Helpers.consts.notes import genNotes

""" 
This code reads audio files from a directory and splits them into 1sec audio segments. 
The segments are then saved to a new directory with a file name that 
includes information about the original file and the segment.
"""

strs = ["E", "A", "D", "G", "B", "ee"]
ds_path = os.path.join(os.getcwd(), "dataset")

_, STRINGS = genNotes(["C2", "A2", "C3", "G3", "C4", "E4"], 22)

print(STRINGS)

isDist = False
isAcoustic = True

for j in range(len(strs)):
    filename = strs[j]
    wav_file_path = f"{ds_path}/strips/{filename}String Acoustic STRIP.wav"

    # CHECK IF THE FILE EXISTS
    if not os.path.isfile(wav_file_path):
        print("File not found: " + wav_file_path)
        continue

    data, samplerate = soundfile.read(wav_file_path)
    index = 0
    for i in range(0, len(data), samplerate):
        # check if the folder exists
        if not os.path.exists(f"{ds_path}/strips/{STRINGS[j][index]}"):
            os.mkdir(f"{ds_path}/strips/{STRINGS[j][index]}")

        # print(STRINGS[j][index])

        add = ""
        if isDist:
            add += "_dist"
        if isAcoustic:
            add += "_acoustic"
        soundfile.write(
            f"{ds_path}/strips/{STRINGS[j][index]}/string-{strs[j]}_fret-{index}_{STRINGS[j][index]}{add}.wav",
            data[i : i + samplerate],
            samplerate,
            subtype="PCM_16",
        )
        """  print("Converted: " + wav_file_path) """
        index += 1
