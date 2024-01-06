from utils.audio.load_prepare import Prepare
import numpy as np
from keras.models import load_model
from midi2audio import FluidSynth
import librosa
from midiutil import MIDIFile
from pydub import AudioSegment
import os

""" 
check the overall points of score system
"""

LABELS = np.load("all_labels.npy")
# mix_model = load_model("Models/model_mix.h5")
model_chords = load_model("Models/model_chords.h5")
model_notes = load_model("Models/model_notes.h5")
model_mix = load_model("Models/model_mix.h5")


def predict_notes(audio_window, model):
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

    AUDIO = audio_window.reshape(1, audio_window.shape[0], audio_window.shape[1], 1)
    predict_notes = model.predict(AUDIO, verbose=0)
    notes_preds, general_confidence = predict(predict_notes)

    return notes_preds, general_confidence


def create_midi_from_array(track_arrays, output_midi_filename, output_wav_filename):
    midi = MIDIFile(len(track_arrays), eventtime_is_ticks=True)  # Number of tracks

    for track_num, note_array in enumerate(track_arrays):
        track = track_num
        time = 0
        tempo = 120  # Adjust the tempo as needed

        midi.addTempo(track, time, tempo)

        for note, sec_in, sec_out in note_array:
            pitch = note  # MIDI note number
            start_time = int(sec_in * 2500)  # Convert seconds to milliseconds
            end_time = int(sec_out * 2500)
            duration = end_time - start_time

            midi.addNote(track, 0, pitch, start_time, duration, 100)  # 100 is velocity

    # Save MIDI file
    with open(output_midi_filename, "wb") as midi_file:
        midi.writeFile(midi_file)

    # Convert MIDI to WAV using FluidSynth
    soundfont_path = (
        "historic/FluidR3_GM.sf2"  # Replace with the path to your SoundFont file
    )
    output_wav_path = "AWA_predict_audio.wav"

    fs = FluidSynth(soundfont_path)
    fs.midi_to_audio(output_midi_filename, output_wav_path)

    # delete the midi file
    os.remove(output_midi_filename)


tracks = []  # Example notes and times in seconds
for i in range(6):
    tracks.append([])


def AWA(
    AUDIO,
    SR,
    ONSETS,
    MaxSteps=None,
    model="chords" or "notes" or "mix",
    gen_audio=False,
):
    ONSETS_SEC, ONSETS_SR = ONSETS
    AUDIO_SECS = len(AUDIO) / SR
    AUDIO_WIN_ACCOMULATE_LEN = int(0.1 * SR)
    AUDIO_WIN_ACCOMULATE_LEN_SEC = 0.1  # 0.1 seconds
    """ The AUDIO_WIN_ACCOMULATE_LEN is the amount of audio that 
    will be acomulated over time with the new audio batches coming from the websocket"""
    MAX_AUDIO_WINDOW_SIZE = int(2.5 * SR)  # 2.5 seconds
    MAX_AUDIO_WINDOW_SIZE_SEC = 2.5  # 2.5 seconds

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

            audio_window = Prepare(
                ad,
                sample_rate=SR,
                pad=True,
            )
            model_to_use = model_chords
            if model == "notes":
                model_to_use = model_notes
            """ elif model == "mix":
                model_to_use = model_mix """
            notes_preds, general_confidence = predict_notes(audio_window, model_to_use)

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

            print(f"|| {best_pred[2]:.2f} - {best_pred[3]:.2f} ||")
            print(f"|| {best_pred[1]:.2f}% ||")

            index = 0

            for note, confidence in best_pred[0]:
                nt = librosa.note_to_midi(note)
                timeIn = best_pred[2]
                timeOut = best_pred[3]

                print(f"|| {note} - {confidence:.2f}% ||")

                tracks[index].append(
                    (
                        nt,
                        timeIn,
                        timeOut,
                    )
                )

                index += 1

        preds += 1

    # if temp folder does not exist, create it
    if not os.path.exists("Temp"):
        os.makedirs("Temp")

    create_midi_from_array(tracks, "Temp/output.mid", "AWA_predict_audio.wav")
