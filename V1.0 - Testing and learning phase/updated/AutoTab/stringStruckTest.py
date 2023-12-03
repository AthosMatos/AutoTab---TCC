from SS_consts import playin_class
from keras.models import load_model
import numpy as np
from audioUtils import loadAndPrepare
import os

model = load_model('stringStruckModel.h5')

interval = 0.02
win = 0.2
winload = 0
windowStart = 0
windowEnd = interval
os.system('cls')
# while windowEnd <= 28:
histogram = []
porcCompleted = 0

while windowEnd <= 8:
    audio, _ = loadAndPrepare(os.path.dirname(
        __file__) + '/dataset/musics/beach house - clean.wav', (windowStart, windowEnd))
    y_pred = model.predict(audio.reshape(1, audio.shape[0], audio.shape[1]))

    for index, pred in enumerate(y_pred[0]):
        print('windowStart: {:.2f} windowEnd: {:.2f} class: {} confidence: {:.2f}%'.format(
            windowStart, windowEnd, playin_class[index], pred*100))
        if (playin_class[index] == 'pluck' and pred > 0.8):

            c = {
                'windowStart': '{:.2f}'.format(windowStart),
                'windowEnd': '{:.2f}'.format(windowEnd),
                'confidence': '{:.2f}'.format(pred*100),
                'class': playin_class[index]
            }
            histogram.append(c)

    if (windowEnd >= win):
        windowStart += interval
    windowEnd += interval
    os.system('cls')
    print('Completed: {:.2f}%'.format(porcCompleted))
    porcCompleted += 100/((win-interval)/interval)

for c in histogram:
    print(c)
