import numpy as np
from utils.audio.load_prepare_prev import load
from utils.audio.rmsNorm import rmsNorm
import librosa.display
import matplotlib.pyplot as plt
from utils.paths import RAW_DATASETS
from historic.getOnsets import get_onsets
from historic.onsetToAudios import onsets_to_audio

fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
# MUSICTESTPATH = RAW_DATASETS.path + "/musics/beach house - clean.wav"
# "/dataset/musics/beach house - clean.wav"
MUSICTESTPATH = RAW_DATASETS.path + "/musics/my bron-yr-aur.mp3"
# MUSICTESTPATH = RAW_DATASETS.path + "/musics/fastnotestSeq.wav"
# MUSICTESTPATH = RAW_DATASETS.path + "/musics/riff test 3 notes.wav"
ad, sr = load(MUSICTESTPATH, seconds_limit=(0, 16))

S = np.abs(librosa.stft(y=ad))

onset_strenght = librosa.onset.onset_strength(S=S, sr=sr)
onset_times = librosa.times_like(onset_strenght, sr=sr)
onset_raw = librosa.onset.onset_detect(onset_envelope=onset_strenght, sr=sr)
rms = librosa.feature.rms(S=S)
onset_bt = librosa.onset.onset_backtrack(onset_raw, rms[0])

# audio_cqt_spec = librosa.amplitude_to_db(audio_cqt_spec, ref=np.max)
# audio_cqt_spec = minmax_scale(audio_cqt_spec)

librosa.display.specshow(
    S,
    y_axis="cqt_hz",
    x_axis="time",
    sr=sr,
    ax=ax[0],
)
ax[0].label_outer()
ax[1].plot(onset_times, onset_strenght, label="Onset strength")
ax[1].vlines(
    librosa.frames_to_time(onset_raw, sr=sr),
    0,
    onset_strenght.max(),
    label="Raw onsets",
    color="g",
    linewidth=2,
    # linestyle="--",
)

ax[1].legend()
ax[1].label_outer()


_, ONSETS = get_onsets(
    ad,
    sr,
)
# _, ONSETS = get_onsets(AUDIO, sr)
onsets_to_audio(ad, ONSETS, sr)

plt.show()
