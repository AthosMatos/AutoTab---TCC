from utils.audio.load_prepare import load
from utils.paths import RAW_DATASETS
from historic.getOnsets import get_onsets
from historic.AudioWindow import audio_window
from keras.models import load_model


""" 
obs:
misssing training data of harmonics in the high frets
"""


def main():
    """model = buildModel(
        train_ds=train_ds,
        val_ds=val_ds,
        TRAIN_Y=TRAIN_Y,
        VAL_Y=VAL_Y,
        INPUT_SHAPE=INPUT_SHAPE,
        NUM_LABELS=NUM_LABELS,
        fit=False,
        saveModel=False,
        savePlot=True,
        summary=True,
    )"""
    MUSICTESTPATH = RAW_DATASETS.path + "/musics/beach house - clean.wav"
    # MUSICTESTPATH = RAW_DATASETS.path + "/musics/riffs test.wav"
    AUDIO, SR = load(MUSICTESTPATH, seconds_limit=(0, 15))
    ONSETS_SECS, ONSETS_SRS = get_onsets(AUDIO, SR)
    # model = load_model("Models/model-out-6-Adam-bigDS.h5")
    model = load_model("Models/model.h5")
    audio_window(AUDIO, (ONSETS_SECS, ONSETS_SRS), SR, model, MaxSteps=20)


if __name__ == "__main__":
    main()

""" 

|| 0 ||
|| 0.09 - 0.29 ||
|| D3 - 99.94% ||

|| 1 ||
|| 0.29 - 0.49 ||
|| G3 - 99.93% ||

|| 1 ||
|| 0.29 - 0.59 ||
|| G3 - 100.00% ||

|| 2 ||
|| 0.57 - 0.77 ||
|| A3 - 91.85% ||

|| 3 ||
|| 0.74 - 0.94 ||
|| C4 - 99.98% ||

|| 3 ||
|| 0.74 - 1.04 ||
|| C4 - 99.98% ||

|| 3 ||
|| 0.74 - 1.14 ||
|| C4 - 99.99% ||

|| 4 ||
|| 1.08 - 1.28 ||
|| A3 - 93.32% ||

|| 4 ||
|| 1.08 - 1.38 ||
|| A3 - 98.95% ||

|| 4 ||
|| 1.08 - 1.48 ||
|| A3 - 99.19% ||

|| 5 ||
|| 1.47 - 1.57 ||
|| C4 - 83.84% ||

|| 5 ||
|| 1.47 - 1.67 ||
|| C4 - 99.94% ||

|| 5 ||
|| 1.47 - 1.77 ||
|| C4 - 96.64% ||

|| 6 ||
|| 1.70 - 1.80 ||
|| D4 - 81.07% ||

|| 6 ||
|| 1.70 - 1.90 ||
|| D4 - 99.99% ||

|| 7 ||
|| 1.83 - 1.93 ||
|| E4 - 96.58% ||

|| 7 ||
|| 1.83 - 2.03 ||
|| E4 - 99.99% ||

|| 7 ||
|| 1.83 - 2.13 ||
|| E4 - 100.00% ||

|| 7 ||
|| 1.83 - 2.23 ||
|| E4 - 99.86% ||

|| 8 ||
|| 2.15 - 2.25 ||
|| G4 - 95.08% ||

|| 8 ||
|| 2.15 - 2.35 ||
|| G4 - 99.98% ||

|| 8 ||
|| 2.15 - 2.45 ||
|| G4 - 100.00% ||

|| 8 ||
|| 2.15 - 2.55 ||
|| G4 - 99.94% ||

|| 9 ||
|| 2.51 - 2.71 ||
|| C4 - 99.99% ||

|| 9 ||
|| 2.51 - 2.81 ||
|| C4 - 99.69% ||

|| 9 ||
|| 2.51 - 2.91 ||
|| C4 - 99.97% ||

|| 9 ||
|| 2.51 - 3.01 ||
|| C4 - 100.00% ||

|| 10 ||
|| 2.98 - 3.18 ||
|| D4 - 99.96% ||

|| 10 ||
|| 2.98 - 3.28 ||
|| D4 - 99.98% ||

|| 10 ||
|| 2.98 - 3.38 ||
|| D4 - 99.99% ||

|| 10 ||
|| 2.98 - 3.48 ||
|| D4 - 100.00% ||

|| 11 ||
|| 3.42 - 3.62 ||
|| D4 - 99.99% ||

|| 12 ||
|| 3.62 - 3.72 ||
|| D#4 - 85.31% ||

|| 12 ||
|| 3.62 - 3.92 ||
|| D#4 - 93.45% ||

|| 13 ||
|| 3.77 - 3.97 ||
|| E4 - 99.81% ||

|| 14 ||
|| 3.95 - 4.35 ||
|| A3 - 92.63% ||

|| 15 ||
|| 4.06 - 4.26 ||
|| A3 - 99.27% ||
|| C#4 - 88.83% ||

|| 15 ||
|| 4.06 - 4.36 ||
|| A3 - 99.87% ||

|| 15 ||
|| 4.06 - 4.46 ||
|| A3 - 99.92% ||

|| 16 ||
|| 4.38 - 4.48 ||
|| C4 - 82.31% ||

|| 16 ||
|| 4.38 - 4.58 ||
|| C4 - 100.00% ||

|| 16 ||
|| 4.38 - 4.68 ||
|| C4 - 99.94% ||

|| 16 ||
|| 4.38 - 4.78 ||
|| C4 - 99.99% ||

|| 16 ||
|| 4.38 - 4.88 ||
|| C4 - 99.99% ||

|| 16 ||
|| 4.38 - 4.98 ||
|| C4 - 99.97% ||

|| 16 ||
|| 4.38 - 5.00 ||
|| C4 - 99.93% ||

"""
