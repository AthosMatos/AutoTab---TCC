import librosa
import numpy as np
import matplotlib.pyplot as plt
from audioUtils3 import load, Prepare

# 'dataset/NoteChangeTrain/Pluck - fender - clean.wav'
# 'dataset/NoteChangeTrain/Ringing - fender - clean.wav'

audio, sr = load(
    'dataset/NoteChangeTrain/Pluck - fender - clean.wav')
audio2, sr2 = load(
    'dataset/NoteChangeTrain/Ringing - fender - clean.wav')

spec = Prepare(audio[0:int(sr/2)], sr)
spec2 = Prepare(audio2[0:int(sr2/2)], sr2)


fig, axis = plt.subplots(2, 1, figsize=(10, 10))
axis[0].set_title('pluck')
axis[1].set_title('running')

axis[0].set_ylabel('Frequency [Hz]')
axis[1].set_ylabel('Frequency [Hz]')
axis[1].set_xlabel('Time [sec]')
axis[0].set_xlabel('Time [sec]')

# plot mfcc
librosa.display.specshow(spec, sr=sr, x_axis='time',
                         y_axis='mel', ax=axis[0], fmax=8000)

librosa.display.specshow(spec2, sr=sr, x_axis='time',
                         y_axis='mel', ax=axis[1], fmax=8000)

plt.show()
