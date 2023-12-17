from utils.audio.load_prepare import Prepare
import numpy as np
from keras.losses import BinaryCrossentropy, BinaryFocalCrossentropy

""" 
after 60% of time btween onsets gone through, start to predict the notes and save the ones with the higher overall confidence
"""


def predict_notes(MODEL, audio_window, LABELS):
    MODEL_PREDICT = MODEL.predict(
        audio_window.reshape(1, audio_window.shape[0], audio_window.shape[1], 1),
        verbose=0,
    )

    notes_preds = []
    sum_of_confidence = 0
    for i, pred in enumerate(MODEL_PREDICT):
        confidence = np.max(pred) * 100
        if confidence > 30:
            note = LABELS[i]
            sum_of_confidence += confidence
            "save the notes"
            notes_preds.append((note, confidence))

    len_of_preds = len(notes_preds)
    if len_of_preds == 0:
        len_of_preds = 1
    "get the sum of all confiedences"
    sum_of_confidence = (sum_of_confidence * 100) / (len_of_preds * 100)

    return notes_preds, sum_of_confidence


def audio_window(AUDIO, ONSETS, SR, MODEL, LABELS, MaxSteps=None, transpose=True):
    ONSETS_SEC, ONSETS_SR = ONSETS
    AUDIO_WIN_ACCOMULATE_LEN = int(0.1 * SR)
    AUDIO_WIN_ACCOMULATE_LEN_SEC = 0.1  # 0.1 seconds
    """ The AUDIO_WIN_ACCOMULATE_LEN is the amount of audio that 
    will be acomulated over time with the new audio batches coming from the websocket"""
    MAX_AUDIO_WINDOW_SIZE = int(2.5 * SR)  # 2.5 seconds
    MAX_AUDIO_WINDOW_SIZE_SEC = 2.5  # 2.5 seconds

    AUDIO_MAX_SEC = len(AUDIO) / SR
    AUDIO_MAX_SAMPLES = len(AUDIO)
    print(f"|| AUDIO_MAX_SEC: {AUDIO_MAX_SEC} ||")

    preds = 0
    for i in range(len(ONSETS_SR)):
        if preds == MaxSteps:
            break
        ONSET = ONSETS_SR[i]
        audio_w_start = ONSET
        audio_w_end = audio_w_start + AUDIO_WIN_ACCOMULATE_LEN
        audio_w_sec_start = ONSETS_SEC[i]
        audio_w_sec_end = audio_w_sec_start + AUDIO_WIN_ACCOMULATE_LEN_SEC

        audio_acomulate = 0
        limitHit = False

        while audio_acomulate < MAX_AUDIO_WINDOW_SIZE_SEC:
            if i < len(ONSETS_SR) - 1:
                if audio_w_sec_end >= ONSETS_SEC[i + 1]:
                    limitHit = True
            else:
                if audio_w_sec_end >= AUDIO_MAX_SEC:
                    limitHit = True
            if audio_w_sec_end > AUDIO_MAX_SEC:
                audio_w_sec_end = AUDIO_MAX_SEC
                audio_w_end = int(AUDIO_MAX_SEC * SR)
                limitHit = True

            audio_window = Prepare(
                AUDIO[audio_w_start:audio_w_end],
                sample_rate=SR,
                transpose=transpose,
            )
            notes_preds, general_confidence = predict_notes(MODEL, audio_window, LABELS)

            if len(notes_preds) == 0:
                audio_w_end += AUDIO_WIN_ACCOMULATE_LEN
                audio_w_sec_end += AUDIO_WIN_ACCOMULATE_LEN_SEC
                audio_acomulate += AUDIO_WIN_ACCOMULATE_LEN_SEC
                continue

            print()
            print(f"|| {preds} ||")
            print(f"|| {audio_w_sec_start:.2f} - {audio_w_sec_end:.2f} ||")

            print(f"|| {general_confidence:.2f}% ||")
            for note, confidence in notes_preds:
                print(f"|| {note} - {confidence:.2f}% ||")

            if limitHit == True:
                break

            audio_w_end += AUDIO_WIN_ACCOMULATE_LEN
            audio_w_sec_end += AUDIO_WIN_ACCOMULATE_LEN_SEC
            audio_acomulate += AUDIO_WIN_ACCOMULATE_LEN_SEC

        preds += 1
