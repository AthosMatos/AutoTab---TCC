import numpy as np
from utils.audio.rmsNorm import rmsNorm
import librosa


def stft_onsets(AUDIO, SR):
    STFT_SPEC = np.abs(librosa.stft(y=AUDIO))

    RMS = librosa.feature.rms(y=AUDIO)
    ONSET_STRENGHT = librosa.onset.onset_strength(S=STFT_SPEC, sr=SR)
    ONSET_RAW = librosa.onset.onset_detect(
        onset_envelope=ONSET_STRENGHT, backtrack=False, sr=SR
    )
    ONSET_BCK_RMS = librosa.onset.onset_backtrack(ONSET_RAW, RMS[0])
    ONSET_FRAMES = librosa.frames_to_time(ONSET_BCK_RMS, sr=SR)

    return ONSET_FRAMES


""" 
NOT AS GOOD AS STFT WITH RMS BACKTRACK
def cqt_onsets(AUDIO, SR):
    CQT_SPEC = np.abs(
        librosa.cqt(
            y=AUDIO,
            sr=SR,
            fmin=librosa.note_to_hz("C2"),
        )
    )
    onset_strenght = librosa.onset.onset_strength(S=CQT_SPEC, sr=SR)
    # onset_times = librosa.times_like(onset_strenght, sr=SR)
    onset_raw = librosa.onset.onset_detect(
        onset_envelope=onset_strenght, backtrack=True, sr=SR
    )
    ONSET_FRAMES = librosa.frames_to_time(onset_raw, sr=SR)

    return ONSET_FRAMES """


def get_onsets(AUDIO, SR):
    AUDIO = rmsNorm(AUDIO, -50)

    ONSET_FRAMES = stft_onsets(AUDIO, SR)
    # ONSET_FRAMES = cqt_onsets(AUDIO, SR)
    ONSET_FRAMES_AS_SAMPLE_RATE = []
    MIN_AUDIO_WINDOW_SIZE = 0.1  # 0.1 seconds / 100ms

    ON_FRAMES = []
    for i in range(len(ONSET_FRAMES) - 1):
        if ONSET_FRAMES[i + 1] - ONSET_FRAMES[i] >= MIN_AUDIO_WINDOW_SIZE:
            ON_FRAMES.append(ONSET_FRAMES[i])
            ONSET_FRAMES_AS_SAMPLE_RATE.append(int(ONSET_FRAMES[i] * SR))
    ONSET_FRAMES_AS_SAMPLE_RATE = np.array(ONSET_FRAMES_AS_SAMPLE_RATE)
    ON_FRAMES = np.array(ON_FRAMES)

    return ON_FRAMES, ONSET_FRAMES_AS_SAMPLE_RATE
