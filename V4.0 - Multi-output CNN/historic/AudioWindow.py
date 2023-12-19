from utils.audio.load_prepare import Prepare
import numpy as np
from keras.models import load_model

""" 
check the overall points of score system
"""


LABELS = np.load("all_labels.npy")
note_model = load_model("Models/model_notes.h5")
chord_model = load_model("Models/model_chords.h5")


def predict_notes(audio_window_chord, audio_window_notes, justNotes, justChords):
    def predict(predct):
        notes_preds = []
        sum_of_confidence = 0
        for i, pred in enumerate(predct):
            confidence = np.max(pred) * 100
            if confidence > 50:
                note = LABELS[i]
                "save the notes"
                notes_preds.append((note, confidence))
                sum_of_confidence += confidence

        len_of_preds = len(notes_preds)
        if len_of_preds == 0:
            len_of_preds = 1
        "get the sum of all confiedences"
        sum_of_confidence = (sum_of_confidence * 100) / (6 * 100)
        return notes_preds, sum_of_confidence

    if justNotes or justChords:
        if justNotes:
            NOTE = audio_window_notes.reshape(
                1, audio_window_notes.shape[0], audio_window_notes.shape[1], 1
            )
            predict_notes = note_model.predict(NOTE, verbose=0)
            notes_preds, general_confidence = predict(predict_notes)
            return notes_preds, general_confidence, True
        if justChords:
            CHORD = audio_window_chord.reshape(
                1, audio_window_chord.shape[0], audio_window_chord.shape[1], 1
            )
            predict_chord = chord_model.predict(CHORD, verbose=0)
            notes_preds_chord, general_confidence_chord = predict(predict_chord)
            return notes_preds_chord, general_confidence_chord, False

    NOTE = audio_window_notes.reshape(
        1, audio_window_notes.shape[0], audio_window_notes.shape[1], 1
    )
    predict_notes = note_model.predict(NOTE, verbose=0)
    notes_preds, general_confidence = predict(predict_notes)
    CHORD = audio_window_chord.reshape(
        1, audio_window_chord.shape[0], audio_window_chord.shape[1], 1
    )
    predict_chord = chord_model.predict(CHORD, verbose=0)
    notes_preds_chord, general_confidence_chord = predict(predict_chord)

    if general_confidence > general_confidence_chord:
        return notes_preds, general_confidence, True

    return notes_preds_chord, general_confidence_chord, False


def audio_window(
    AUDIO,
    SR,
    ONSETS,
    MaxSteps=None,
    giveMoreAudioContext=False,
    justNotes=False,
    justChords=False,
):
    ONSETS_SEC, ONSETS_SR = ONSETS
    AUDIO_SECS = len(AUDIO) / SR
    AUDIO_WIN_ACCOMULATE_LEN = int(0.1 * SR)
    AUDIO_WIN_ACCOMULATE_LEN_SEC = 0.1  # 0.1 seconds
    """ The AUDIO_WIN_ACCOMULATE_LEN is the amount of audio that 
    will be acomulated over time with the new audio batches coming from the websocket"""
    MAX_AUDIO_WINDOW_SIZE = int(SR)  # 2.5 seconds
    MAX_AUDIO_WINDOW_SIZE_SEC = 1  # 2.5 seconds

    AUDIO_MAX_SEC = AUDIO_SECS
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

        best_pred = None

        while audio_acomulate < MAX_AUDIO_WINDOW_SIZE_SEC:
            """if audio_acomulate >= 0.20:
            isNotes = False
            MODEL = chord_model"""

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

            ad = AUDIO[audio_w_start:audio_w_end]
            if giveMoreAudioContext:
                ad = AUDIO[audio_w_start : audio_w_end + int(SR * 0.6)]
            audio_window_chord = Prepare(
                ad,
                # audio_limit_sec=(audio_w_sec_start, audio_w_sec_end),
                sample_rate=SR,
                notes=False,
            )
            audio_window_note = Prepare(
                ad,
                # audio_limit_sec=(audio_w_sec_start, audio_w_sec_end),
                sample_rate=SR,
                notes=True,
            )
            notes_preds, general_confidence, isNotes = predict_notes(
                audio_window_chord, audio_window_note, justNotes, justChords
            )

            if len(notes_preds) == 0:
                audio_w_end += AUDIO_WIN_ACCOMULATE_LEN
                audio_w_sec_end += AUDIO_WIN_ACCOMULATE_LEN_SEC
                audio_acomulate += AUDIO_WIN_ACCOMULATE_LEN_SEC
                continue

            if best_pred == None or general_confidence > best_pred[1]:
                best_pred = (
                    notes_preds,
                    general_confidence,
                    audio_w_sec_start,
                    audio_w_sec_end,
                    isNotes,
                )

            """ print()
            print(f"|| {preds} ||")
            if giveMoreAudioContext:
                print(f"|| {audio_w_sec_start:.2f} - {audio_w_sec_end+0.6:.2f} ||")
            else:
                print(f"|| {audio_w_sec_start:.2f} - {audio_w_sec_end:.2f} ||")
            print(f"|| {general_confidence:.2f}% ||")
            for note, confidence in notes_preds:
                print(f"|| {note} - {confidence:.2f}% ||") """

            if limitHit == True:
                break

            audio_w_end += AUDIO_WIN_ACCOMULATE_LEN
            audio_w_sec_end += AUDIO_WIN_ACCOMULATE_LEN_SEC
            audio_acomulate += AUDIO_WIN_ACCOMULATE_LEN_SEC
        if best_pred != None:
            print()
            print(f"|| {preds} ||")
            if giveMoreAudioContext:
                print(f"|| {best_pred[2]:.2f} - {best_pred[3]+0.6:.2f} ||")
            else:
                print(f"|| {best_pred[2]:.2f} - {best_pred[3]:.2f} ||")
            if best_pred[4]:
                print(f"|| Notes ||")
            else:
                print(f"|| Chords ||")
            print(f"|| {best_pred[1]:.2f}% ||")

            for note, confidence in best_pred[0]:
                print(f"|| {note} - {confidence:.2f}% ||")

        preds += 1
