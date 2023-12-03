from prepareTrain import audio_inputs, notes_class
from keras.models import load_model
import numpy as np
from audioUtils import load, Prepare
import os
import librosa

model = load_model('NotesNeuralNet.h5')

interval = 0.05
maxWIN = 0.5
win = interval
winload = 0
windowStart = 0
windowEnd = interval
os.system('cls')
# while windowEnd <= 28:
histogram = []
porcCompleted = 0
maxSecnds = 8

while windowEnd <= maxSecnds:

    audio, sr = load(os.path.dirname(
        __file__) + '/dataset/musics/beach house - clean.wav', (windowStart, windowEnd))
    o_env = librosa.onset.onset_strength(y=audio, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)

    spec, _ = Prepare(audio, sr)

    y_pred = model.predict(spec.reshape(1, spec.shape[0], spec.shape[1]))
    best_pred = np.argmax(y_pred)
    confidence = np.max(y_pred)*100

    if win < maxWIN:
        win += interval
    else:
        windowStart += interval
    windowEnd += interval

    # if found more than one onset, cut the window to end of the window
    if (notes_class[best_pred] != 'noise'):
        c = {
            'windowStart': '{:.2f}'.format(windowStart),
            'windowEnd': '{:.2f}'.format(windowEnd),
            'confidence': '{:.2f}'.format(confidence),
            'class': notes_class[best_pred],
            'onsets': len(onset_frames)
        }
        histogram.append(c)

        win = interval
        windowStart = windowEnd
        windowEnd += win

    """ if (len(onset_frames) >= 1 and notes_class[best_pred] != 'noise'):
        win = interval
        windowStart = windowEnd
        windowEnd += win
    else: 
        if (len(onset_frames) > 1):  # if found more than one onset, cut the window to end of the window
            win = interval
            windowStart = windowEnd
            windowEnd += win
        else:
            if win < maxWIN:
                win += interval
            else:
                windowStart += interval

            windowEnd += interval """

    os.system('cls')
    print('Completed: {:.2f}%'.format(porcCompleted))
    porcCompleted = (100 * windowEnd)/maxSecnds

for c in histogram:
    print(c)



""" 
Params
MaxWindow = 0.5s
This define the max seconds (amount of data) of audio that is going to be analysed
MinWindow = 0.2s
This define the min seconds (amount of data) of audio that is going to be analysed
WindowStep = 0.02s
This define the amount of seconds that the window is going to move



"""