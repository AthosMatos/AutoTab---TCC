from utils.audio.load_prepare import load
from utils.paths import RAW_DATASETS
from historic.getOnsets import get_onsets
from historic.AudioWindow import audio_window
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

""" 
obs:
misssing training data of harmonics in the high frets
record some power chords and aplly data aug to them
Athosset has some audios that end and the file continues with silence, remove it
"""

sr = 44100

LABELS = np.load("all_labels.npy")


def main():
    MUSICTESTPATH = RAW_DATASETS.path + "/musics/g chord.wav"
    # MUSICTESTPATH = RAW_DATASETS.path + "/musics/beach house - clean.wav"
    # MUSICTESTPATH = RAW_DATASETS.path + "/musics/fastnotestSeq.wav"
    # MUSICTESTPATH = RAW_DATASETS.path + "/musics/riffs test.wav"

    """
    riff test notes in riff
    1- A2 E3 A3
    2- D3 A3 D4
    3 - E3 B3 E4

    """

    AUDIO, SR = load(MUSICTESTPATH, seconds_limit=(0, 15), sample_rate=sr)
    # print(AUDIO)
    """ plt.plot(AUDIO)
    plt.show() """
    ONSETS_SECS, ONSETS_SRS = get_onsets(AUDIO, SR)
    # model = load_model("Models/model-out-6-Adam-bigDS.h5")
    model = load_model("Models/model_chords.h5")
    # model = load_model("Models/bestModel_stft_16k.h5")
    audio_window(
        AUDIO,
        (ONSETS_SECS, ONSETS_SRS),
        SR,
        model,
        LABELS=LABELS,
        MaxSteps=20,
        transpose=False,
    )


if __name__ == "__main__":
    main()
