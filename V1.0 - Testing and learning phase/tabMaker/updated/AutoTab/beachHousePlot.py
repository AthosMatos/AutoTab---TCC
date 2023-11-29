from utils import rmsNorm, hop_length, nftt, loadAndPrepare
from librosa.display import specshow
import matplotlib.pyplot as plt
import os


""" S, sample_rate = loadAndPrepare(os.path.dirname(
    __file__) + '/dataset/distortion - audio/D3.wav')
S2, _ = loadAndPrepare(os.path.dirname(__file__) +
                       '/dataset/distortion - audio/E3.wav')
S3, _ = loadAndPrepare(os.path.dirname(__file__) +
                       '/dataset/distortion - audio/G3.wav') """


S4, sample_rate = loadAndPrepare(os.path.dirname(__file__) +
                                 '/Tests/audio/beach house.wav', audio_limit_sec=(0, 2))


print(S4.shape)

# S[S < int(-56/2)] = np.min(S)
# S = minmax_scale(S, axis=0)
# notes interval = 0.5sec
""" axis[0].set_title('D3')
specshow(S, x_axis='time', y_axis='log', sr=sample_rate,
         ax=axis[0], n_fft=nftt, hop_length=hop_length)

axis[1].set_title('E3')
specshow(S2, x_axis='time', y_axis='log', sr=sample_rate,
         ax=axis[1], n_fft=nftt, hop_length=hop_length)

axis[2].set_title('G3')
specshow(S3, x_axis='time', y_axis='log', sr=sample_rate,
         ax=axis[2], n_fft=nftt, hop_length=hop_length) """
specshow(S4, x_axis='time', y_axis='mel', fmax=8000,)
plt.title('beach house - NOTES')

plt.show()
