import os
from keras.utils import to_categorical
import numpy as np
import json
from labels import LABELS
import numpy as np
from vars import MAX_AUDIO_EVENTS


def gen_Y():
    notes = []
    times = []

    # index each label - one line
    lbls = [v for v, _ in enumerate(LABELS)]
    hot = to_categorical(lbls, num_classes=len(LABELS))
    # LABELS_SIZE = len(LABELS)

    MAX_SIMUTANEOUS_NOTES = 6
    # load JSON file
    with open(os.path.dirname(__file__) + "/newDS/beachHouse/data.json", "r") as f:
        data = json.load(f)
        for key in data:
            # print(data[key])
            nts = []
            t = []
            for i, dt in enumerate(data[key]):
                for j, d in enumerate(dt):
                    index = LABELS.index(d)
                    nts.append(hot[index])
                    # nts.append(lbls[index])
                    t.append(i)
                    # print(d, index)
            if len(nts) < MAX_AUDIO_EVENTS:
                for i in range(MAX_AUDIO_EVENTS - len(nts)):
                    nts.append(hot[-1])
                    t.append(-1)
            notes.append(nts)
            times.append(t)
        """ if len(dd) < MAX_SIMUTANEOUS_NOTES:
                    for i in range(6 - len(dd)):
                        dd.append(hot[-1]) """

        """ notes.append(dd) """
        """  if len(yy) < MAX_AUDIO_EVENTS:
                for i in range(MAX_AUDIO_EVENTS - len(yy)):
                    temp = []
                    for j in range(MAX_SIMUTANEOUS_NOTES):
                        temp.append(hot[-1])
                    yy.append(temp) """
        """  yy = np.array(yy)
            y.append(yy) """

    """ y = np.array(y) """
    notes = np.array(notes).reshape(1, len(notes), MAX_AUDIO_EVENTS, len(LABELS))
    times = np.array(times).reshape(1, len(times), MAX_AUDIO_EVENTS)
    print(f"Ytrain notes shape{notes.shape}")
    print(f"Ytrain times shape{times.shape}")
    """ for i, d in enumerate(notes):
        print(d.shape)
        print(times[i]) """
    """ flatten = []
    for i,ex in enumerate(y):
        print(ex.shape,i)
        print(ex.shape)
        #flatten.append(len(ex.reshape(ex.shape[0] * ex.shape[1] * ex.shape[2])))
    flatten = [flatten] """
    """ shape (TIMESTEPS,MAXAUDIOEVENTS,SIMULTANEOSNOTES,CATEGORIES) """
    """ print(f"Time steps = {y.shape[0]}")
    print(f"Audio events = {y.shape[1]}")
    print(f"Max. Simultaneous Notes = {y.shape[2]}")
    print(f"Notes categories = {y.shape[3]}") """

    # Y = y.reshape(1, y.shape[0], y.shape[1] * y.shape[2] * y.shape[3])
    # print(Y.shape)
    """ y = np.expand_dims(y, 0) """
    return notes, times
