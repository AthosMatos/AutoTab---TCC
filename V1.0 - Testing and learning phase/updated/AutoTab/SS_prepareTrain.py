
import os
import numpy as np
from keras.utils import to_categorical
from SS_consts import playin_class
from audioUtils3 import Prepare, load


def prepareOutput(output, amounClasses):
    output = np.array(output)
    unique_output = np.unique(output)
    category_mapping = {category: i for i,
                        category in enumerate(unique_output)}
    output = to_categorical([category_mapping[category]
                            for category in output], amounClasses)
    return output


audio_inputs = []
outputs = []

# run through all files in dataset
for filename in os.listdir(os.path.dirname(__file__) + '/dataset/NoteChangeTrain'):
    if filename.endswith(".wav"):
        audio, sr = load(os.path.dirname(
            __file__) + '/dataset/NoteChangeTrain/' + filename)
    playS = ''
    # if the filename contains a word in the playin_class list, then it is the class of the audio
    for playin in playin_class:
        # filename to lower case
        if playin in filename.lower():
            playS = playin
            break

    if playS != '':
        # the wav file has many samples with a interval of 0.5s
        for i in range(0, audio.shape[0], int(sr/2)):
            start = i
            end = i+int(sr/2)
            audio_inputs.append(Prepare(audio[start:end], sr))
            outputs.append(playS)


audio_inputs = np.array(audio_inputs)
outputs = prepareOutput(outputs, len(playin_class))

permutation_index = np.random.permutation(len(audio_inputs))
# Shuffle both arrays using the same permutation index
audio_inputs = audio_inputs[permutation_index]
outputs = outputs[permutation_index]

# Print the shuffled arrays
print("audio_inputs:", audio_inputs.shape)
print("outputs:", outputs.shape)

x_train = audio_inputs
y_train = outputs
