import os
import numpy as np
from keras.utils import to_categorical
from Helpers.DataPreparation.audioUtils import loadAndPrepare
from Helpers.consts.paths import training_ds_path
import os

audio_inputs = []
notes_outputs = []
labels = []

amount_of_files = 0
for root, dirs, files in os.walk(training_ds_path):
    for dir in dirs:
        for file in os.listdir(os.path.join(root, dir)):
            if file.endswith("wav"):
                amount_of_files += 1

""" labels_ignore = ["Major", "Minor"] """

for root, dirs, files in os.walk(training_ds_path):
    # use folders as classes notes_outputs
    for dir in dirs:
        # if dir has substring in labels_ignore, ignore it
        """if any(label in dir for label in labels_ignore):
        continue"""
        for file in os.listdir(os.path.join(root, dir)):
            if file.endswith("wav"):
                wav_file_path = os.path.join(root, dir, file)
                S, _ = loadAndPrepare(wav_file_path, (None, None))
                audio_inputs.append(S)
                notes_outputs.append(dir)
                labels.append(dir)

                os.system("cls")
                """ # print folder
                print(f"Folder: {dir}")
                # print filename
                print(f"File: {file}") """
                # print completion porcentage
                print(f"Loading training dataset: {round((len(audio_inputs) / amount_of_files) * 100, 2)}%")


def prepareOutput(output, labels):
    output = np.array(output)
    category_mapping = {category: i for i, category in enumerate(labels)}
    output = to_categorical([category_mapping[category] for category in output])
    return output


# save the x_train and y_train as numpy arrays
audio_inputs = np.array(audio_inputs)
labels = np.unique(labels)
notes_outputs = prepareOutput(notes_outputs, labels)
os.system("cls")
print("audio_inputs: ", audio_inputs.shape)
print("labels: ", labels)
print("notes_outputs: ", notes_outputs.shape)
