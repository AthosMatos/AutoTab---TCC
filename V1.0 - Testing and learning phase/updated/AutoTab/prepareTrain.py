import os
import numpy as np
from keras.utils import to_categorical
from audioUtils import loadAndPrepare
from consts import notes_class, amps_class, playin_class, playin_class2, gain_class, all_classes
import sys
import os

audio_inputs = []
notes_outputs = []
amps_outputs = []
playS_outputs = []
playS2_outputs = []
gain_outputs = []


for root, dirs, files in os.walk(os.path.dirname(__file__) + '/dataset'):
    os.system('cls')
    print('dirs: ', dirs)

    for file in files:
        if file.endswith('wav'):
            wav_file_path = os.path.join(root, file)
            # check if its in all_notes
            filename = file.split('.')[0].split('-')[0].strip()
            if filename in notes_class:
                addedAmps = False
                addedPlayin = False
                addedPlayin2 = False
                addedGain = False

                S, _ = loadAndPrepare(wav_file_path, (None, None))
                audio_inputs.append(S)
                notes_outputs.append(filename)
                for r in root.split('\\'):
                    if (r in all_classes):
                        if r in amps_class:
                            amps_outputs.append(r)
                            addedAmps = True
                        if r in playin_class:
                            playS_outputs.append(r)
                            addedPlayin = True
                        if r in playin_class2:
                            playS2_outputs.append(r)
                            addedPlayin2 = True
                        if r in gain_class:
                            gain_outputs.append(r)
                            addedGain = True
                if not addedAmps:
                    amps_outputs.append('none')
                if not addedPlayin:
                    playS_outputs.append('none')
                if not addedPlayin2:
                    playS2_outputs.append('none')
                if not addedGain:
                    gain_outputs.append('none')


def prepareOutput(output):
    output = np.array(output)
    unique_output = np.unique(output)
    category_mapping = {category: i for i,
                        category in enumerate(unique_output)}
    output = to_categorical([category_mapping[category]
                            for category in output])
    return output


# save the x_train and y_train as numpy arrays
audio_inputs = np.array(audio_inputs)

notes_outputs = prepareOutput(notes_outputs)
amps_outputs = prepareOutput(amps_outputs)
playS_outputs = prepareOutput(playS_outputs)
playS2_outputs = prepareOutput(playS2_outputs)
gain_outputs = prepareOutput(gain_outputs)

os.system('cls')
print('notes_outputs: ', notes_outputs.shape)
print('amps_outputs: ', amps_outputs.shape)
print('playS_outputs: ', playS_outputs.shape)
print('playS2_outputs: ', playS2_outputs.shape)
print('gain_outputs: ', gain_outputs.shape)


""" 
print('unique_pitches: ', unique_pitches)
print('category_mapping: ', category_mapping) """
# print(y_train.argmax(
# print(y_train.argmax(axis=1)) use as index to get the note from all_notes


"""
np.save('x_train.npy', all_batches)
np.save('y_train.npy', y_train) """
