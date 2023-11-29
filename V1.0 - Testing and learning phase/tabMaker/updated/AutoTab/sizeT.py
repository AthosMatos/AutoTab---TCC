import os
import numpy as np
from keras.utils import to_categorical
from audioUtils5 import loadAndPrepare
import sys
import os

audio_inputs = []
notes_outputs = []


""" for file in files:
        if file.endswith("wav"):
            wav_file_path = os.path.join(root, file)
            # check if its in all_notes
            filename = file.split(".")[0].split("-")[0].strip()
            
            print("dirs: ", dirs)
            
            S, _ = loadAndPrepare(wav_file_path, (None, None))
            audio_inputs.append(S)
            notes_outputs.append(filename) """

for root, dirs, files in os.walk(os.path.dirname(__file__) + "/dataset/training"):
    os.system("cls")

    # use folders as classes notes_outputs
    for dir in dirs:
        for file in os.listdir(os.path.join(root, dir)):
            if file.endswith("wav"):
                wav_file_path = os.path.join(root, dir, file)
                S, _ = loadAndPrepare(wav_file_path, (None, None))
                audio_inputs.append(S)
                notes_outputs.append(dir)


def prepareOutput(output):
    output = np.array(output)
    unique_output = np.unique(output)
    category_mapping = {category: i for i, category in enumerate(unique_output)}
    output = to_categorical([category_mapping[category] for category in output])
    return output


# save the x_train and y_train as numpy arrays
audio_inputs = np.array(audio_inputs)

""" notes_outputs = prepareOutput(notes_outputs)


os.system("cls")
print("notes_outputs: ", notes_outputs.shape) """


""" 
print('unique_pitches: ', unique_pitches)
print('category_mapping: ', category_mapping) """
# print(y_train.argmax(
# print(y_train.argmax(axis=1)) use as index to get the note from all_notes


"""
np.save('x_train.npy', all_batches)
np.save('y_train.npy', y_train) """
