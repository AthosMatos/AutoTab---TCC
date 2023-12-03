import os
import numpy as np
from keras.utils import to_categorical
from audioUtils import loadAndPrepare
from consts import all_notes,all_gainLevels,all_amps

# walk through the dataset and process each wav file
# divide each 0.5sec wav file into batches of 0.1sec
audio_inputs = []
notes_outputs = []
gainlvl_outputs = []
amps_outputs = []

for root, dirs, files in os.walk(os.path.dirname(__file__) + '/dataset'):
    os.system('cls')
    print('dirs: ', dirs)
    """ 
    for file in files:
        if file.endswith('wav'):
            wav_file_path = os.path.join(root, file)

            # check if its in all_notes
            filename = file.split('.')[0].split('-')[0].strip()
            if filename in all_notes:
                S, _ = loadAndPrepare(wav_file_path, (None, None))
                audio_inputs.append(S)
                notes_outputs.append(filename) """
                
            
            
            

# save the x_train and y_train as numpy arrays
audio_inputs = np.array(audio_inputs)
unique_pitches = np.unique(notes_outputs)

category_mapping = {category: i for i,
                    category in enumerate(unique_pitches)}


notes_outputs = to_categorical([category_mapping[category]
                          for category in notes_outputs])

os.system('cls')
print(notes_outputs.shape)
print(audio_inputs.shape)

""" 
print('unique_pitches: ', unique_pitches)
print('category_mapping: ', category_mapping) """
# print(y_train.argmax(
# print(y_train.argmax(axis=1)) use as index to get the note from all_notes


"""
np.save('x_train.npy', all_batches)
np.save('y_train.npy', y_train) """
