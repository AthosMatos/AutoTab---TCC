from prepareTrain import audio_inputs, notes_class
from keras.models import load_model
import numpy as np
from audioUtils import loadAndPrepare
import os

model = load_model('NotesNeuralNet.h5')

windowStart = 0.0
windowEnd = 0.2

audio, _ = loadAndPrepare(os.path.dirname(
    __file__) + '/dataset/musics/beach house - clean.wav', (windowStart, windowEnd))
y_pred = model.predict(audio.reshape(1, audio.shape[0], audio.shape[1]))
os.system('cls')
print('|| Audio Window: ', windowStart, ' - ', windowEnd, ' ||')


# print('max: ', "{:.4f}".format(np.max(y_pred)*100), '%')  # confidence
# print('np.argmax(y_pred): ', np.argmax(y_pred))  # index
# print('Predicted note: ', all_notes[np.argmax(y_pred)])  # note
# each pred notes confidende
for index, pred in enumerate(y_pred[0]):
    print(notes_class[index], ': ', "{:.4f}%".format(pred*100))