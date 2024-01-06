import numpy as np
from utils.audio.load_prepare_prev import load
from utils.audio.rmsNorm import rmsNorm
import librosa.display
import matplotlib.pyplot as plt
from utils.paths import RAW_DATASETS
from historic.getOnsets import get_onsets
from historic.onsetToAudios import onsets_to_audio

# MUSICTESTPATH = RAW_DATASETS.path + "/musics/beach house - clean.wav"
# "/dataset/musics/beach house - clean.wav"
# MUSICTESTPATH = RAW_DATASETS.path + "/musics/my bron-yr-aur.mp3"
# MUSICTESTPATH = RAW_DATASETS.path + "/musics/fastnotestSeq.wav"
MUSICTESTPATH = RAW_DATASETS.path + "/musics/simple notes test.wav"
# MUSICTESTPATH = RAW_DATASETS.path + "/musics/riff test 3 notes.wav"
# MUSICTESTPATH = RAW_DATASETS.path + "/musics/riffs test.wav"
ad, sr = load(MUSICTESTPATH, seconds_limit=(0, 16))

D = np.abs(librosa.stft(y=ad))

o_env = librosa.onset.onset_strength(
    y=ad, sr=sr, aggregate=np.median, fmax=8000, n_mels=256
)
times = librosa.times_like(o_env, sr=sr)
onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)

# rms = librosa.feature.rms(S=D)
onset_bt_rms = librosa.onset.onset_backtrack(onset_frames, o_env)


fig, ax = plt.subplots(nrows=2, sharex=True)
librosa.display.specshow(
    librosa.amplitude_to_db(D, ref=np.max), y_axis="log", x_axis="time", ax=ax[0], sr=sr
)
ax[0].label_outer()
ax[1].plot(times, o_env, label="Onset strength")
ax[1].vlines(
    librosa.frames_to_time(onset_frames, sr=sr),
    0,
    o_env.max(),
    label="Raw onsets",
    color="g",
)
ax[1].vlines(
    librosa.frames_to_time(onset_bt_rms, sr=sr),
    0,
    o_env.max(),
    label="Backtracked",
    color="r",
)
ax[1].legend()
ax[1].label_outer()


_, ONSETS = get_onsets(
    ad,
    sr,
    # onset_frames=librosa.frames_to_time(onset_frames, sr=sr),
    # onset_frames=librosa.frames_to_time(onset_bt_rms, sr=sr),
)
# _, ONSETS = get_onsets(AUDIO, sr)
print(ONSETS)
onsets_to_audio(ad, ONSETS, sr)

plt.show()
