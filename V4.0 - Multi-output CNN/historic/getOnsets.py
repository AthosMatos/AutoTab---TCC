import numpy as np
from utils.audio.rmsNorm import rmsNorm
import librosa


def stft_onsets(AUDIO, SR):
    S = np.abs(librosa.stft(y=AUDIO))

    onset_strenght = librosa.onset.onset_strength(S=S, sr=SR)
    onset_raw = librosa.onset.onset_detect(onset_envelope=onset_strenght, sr=SR)
    rms = librosa.feature.rms(S=S)
    onset_bt = librosa.onset.onset_backtrack(onset_raw, rms[0])

    ONSET_FRAMES = librosa.frames_to_time(onset_bt, sr=SR)

    return ONSET_FRAMES


#
def cqt_onsets(AUDIO, SR):
    CQT_SPEC = np.abs(
        librosa.cqt(
            y=AUDIO,
            sr=SR,
            fmin=librosa.note_to_hz("C2"),
        )
    )
    CQT_SPEC = librosa.amplitude_to_db(CQT_SPEC, ref=np.max)

    onset_strenght = librosa.onset.onset_strength(S=CQT_SPEC, sr=SR)
    # onset_times = librosa.times_like(onset_strenght, sr=SR)
    onset_raw = librosa.onset.onset_detect(
        onset_envelope=onset_strenght, backtrack=False, sr=SR
    )
    ONSET_FRAMES = librosa.frames_to_time(onset_raw, sr=SR)

    return ONSET_FRAMES


def get_onsets(AUDIO, SR, onset_frames=None):
    AUDIO = rmsNorm(AUDIO, -50)
    if onset_frames is None:
        ONSET_FRAMES = stft_onsets(AUDIO, SR)
    else:
        ONSET_FRAMES = onset_frames
    # ONSET_FRAMES = cqt_onsets(AUDIO, SR)
    ONSET_FRAMES_AS_SAMPLE_RATE = []
    MIN_AUDIO_WINDOW_SIZE = 0.1  # 0.1 seconds / 100ms

    ON_FRAMES = []
    for i in range(len(ONSET_FRAMES) - 1):
        print(
            f"ONSET_FRAMES[i + 1] - ONSET_FRAMES[i]: {ONSET_FRAMES[i + 1] - ONSET_FRAMES[i]}"
        )
        if ONSET_FRAMES[i + 1] - ONSET_FRAMES[i] >= MIN_AUDIO_WINDOW_SIZE:
            ON_FRAMES.append(ONSET_FRAMES[i])
            ONSET_FRAMES_AS_SAMPLE_RATE.append(int(ONSET_FRAMES[i] * SR))

    ONSET_FRAMES_AS_SAMPLE_RATE = np.array(ONSET_FRAMES_AS_SAMPLE_RATE)
    ON_FRAMES = np.array(ON_FRAMES)

    return ON_FRAMES, ONSET_FRAMES_AS_SAMPLE_RATE
