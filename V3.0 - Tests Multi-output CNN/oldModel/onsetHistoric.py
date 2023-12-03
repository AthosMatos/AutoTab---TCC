import numpy as np
from Helpers.DataPreparation.audioUtils import load, rmsNorm, Prepare
import os
import librosa
from keras.models import load_model
import soundfile as sf
from Helpers.consts.paths import ds_path

model = load_model("BEST-model.keras")
# model = load_model("NotesNeuralNet.h5")
labels = np.load("np_files/unique_labels.npy")

# is like the wesocket channel awaiting for the music to come
""" CHANNELTESTPATH = os.path.dirname(__file__) + "/dataset/musics/my bron-yr-aur.mp3"
RAW_COMPLETE_AUDIO_TEST, SR = load(CHANNELTESTPATH, (0, 7)) """

# is like the wesocket channel awaiting for the music to come
CHANNELTESTPATH = os.path.dirname(__file__) + "/dataset/musics/beach house - clean.wav"
RAW_COMPLETE_AUDIO_TEST, SR = load(CHANNELTESTPATH, (0, 20))
""" 
CHANNELTESTPATH = os.path.dirname(__file__) + "/dataset/musics/TEST SONG.wav"
RAW_COMPLETE_AUDIO_TEST, SR = load(CHANNELTESTPATH) """
""" This is the music coming from the websocket, it acomulates over time """


AUDIO_WIN_ACCOMULATE_LEN = int(0.02 * SR)  # 0.02s - 20ms
""" The AUDIO_WIN_ACCOMULATE_LEN is the amount of audio that 
will be acomulated over time with the new audio batches coming from the websocket"""
MAX_WINDOW_SIZE = int(2.5 * SR)  # 2.5 seconds
audio_win_len = int(0.2 * SR)  # 0.2s - 200ms
audio_w_start = 0  # audio window start
audio_w_end = audio_win_len  # audio window end

sound_hist = []
seconds_elapsed = 0
tab = []
batch_predictions = []

"This part is for the testing of a websocket based ambient"
WEBSOCKET_AUDIO_BATCH_STD_LEN = int(0.5 * SR)  # 0.5s - 500ms
WEBSOCKET_AUDIO_BATCH_s = 0  # websocket audio batch start
WEBSOCKET_AUDIO_BATCH_e = WEBSOCKET_AUDIO_BATCH_STD_LEN  # websocket audio batch end


def onsets(RMS_NORM_RAW_AUDIO_BATCH):
    onset_raw = librosa.onset.onset_detect(
        onset_envelope=librosa.onset.onset_strength(y=RMS_NORM_RAW_AUDIO_BATCH, sr=SR),
        backtrack=False,
        sr=SR,
    )
    rms = librosa.feature.rms(S=np.abs(librosa.stft(y=RMS_NORM_RAW_AUDIO_BATCH)))
    bck_times = np.unique(
        librosa.frames_to_time(librosa.onset.onset_backtrack(onset_raw, rms[0]), sr=SR)
    )
    onsets = []

    for i in range(bck_times.shape[0] - 1):
        onset_start = bck_times[i]
        onsets_end = bck_times[i + 1]
        onsets.append((onset_start, onsets_end))

    return onsets


def on_message():
    RMS_NORM_RAW_AUDIO_BATCH = rmsNorm(np.array(sound_hist), -50)
    # RMS_NORM_RAW_AUDIO_BATCH = np.array(sound_hist)
    """ Normalization to -50db in order to get the a best definition for the detection of the onsets
    get the onsets of the audio batch """
    onsets_times = onsets(RMS_NORM_RAW_AUDIO_BATCH)
    """ get the onsets of the audio batch """
    # sum with the seconds elapsed
    FIX_TIMING_ONSET_TIMES = [
        (x[0] + seconds_elapsed, x[1] + seconds_elapsed) for x in onsets_times
    ]
    if FIX_TIMING_ONSET_TIMES.__len__() == 0:
        return []

    """ onsets with the seconds elapsed """
    print(
        f"|| onsets: from {FIX_TIMING_ONSET_TIMES[0][0]:.2f}s to {FIX_TIMING_ONSET_TIMES[-1][1]:.2f}s ||"
    )
    # print amount of found onsets
    # print(f"|| found {len(onsets_times)} onsets ||")
    # print onsets with 2 decimal places
    preds = []
    for i, onset in enumerate(onsets_times):
        """if onset[1] - onset[0] < 0.1:
        continue"""
        "if the onset window is less than 0.1s, skip it because theres no way to detect the chord or a note in that window"
        TO_PROCESS_AUDIO = sound_hist[
            int(onset[0] * SR) : int(onset[1] * SR)
        ]  # in here we use the onset_times becouse the sound_hist is based on a time window of 2.5s and resets every 2.5s
        "get the audio window to process"
        spec = Prepare(np.array(TO_PROCESS_AUDIO), SR)
        "prepare the audio window for the neural network"
        y_pred = model.predict(spec.reshape(1, spec.shape[0], spec.shape[1]), verbose=0)
        "predict the audio window"
        best_pred = np.argmax(y_pred)
        "get the best prediction"
        confidence = np.max(y_pred) * 100
        "get the confidence of the prediction"

        """ if confidence < 60:
            continue """
        print(
            f"{labels[best_pred]} - {int(confidence)}%" + " | "
            f"{FIX_TIMING_ONSET_TIMES[i][0]:.2f}s - {FIX_TIMING_ONSET_TIMES[i][1]:.2f}s",
        )
        if labels[best_pred] != "noise":
            # preds.append((labels[best_pred], int(confidence)))
            preds.append(
                {
                    "label": labels[best_pred],
                    "confidence": int(confidence),
                    "start": FIX_TIMING_ONSET_TIMES[i][0],
                    "end": FIX_TIMING_ONSET_TIMES[i][1],
                }
            )

    return preds


""" this is going to be the onmessage of the websocket """
end = False
while True:
    if sound_hist.__len__() > MAX_WINDOW_SIZE:
        tab.extend(batch_predictions)
        batch_predictions = []
        sound_hist = []
        seconds_elapsed += MAX_WINDOW_SIZE / SR

    if WEBSOCKET_AUDIO_BATCH_e >= RAW_COMPLETE_AUDIO_TEST.shape[0]:
        WEBSOCKET_AUDIO_BATCH_e = RAW_COMPLETE_AUDIO_TEST.shape[0]
        end = True
    print("\n-------------------------------------------\n")
    print(
        f"Audio acomulated from {seconds_elapsed:.2f}s to {WEBSOCKET_AUDIO_BATCH_e / SR:.2f}s"
    )
    MSG_AUDIO = RAW_COMPLETE_AUDIO_TEST[WEBSOCKET_AUDIO_BATCH_s:WEBSOCKET_AUDIO_BATCH_e]
    sound_hist.extend(MSG_AUDIO)
    batch_predictions = on_message()

    if end:
        tab.extend(batch_predictions)
        print("\n-------------------------------------------\n")
        break

    WEBSOCKET_AUDIO_BATCH_s += WEBSOCKET_AUDIO_BATCH_STD_LEN
    WEBSOCKET_AUDIO_BATCH_e += WEBSOCKET_AUDIO_BATCH_STD_LEN

    """ if WEBSOCKET_AUDIO_BATCH_e > RAW_COMPLETE_AUDIO_TEST.shape[0]:
        WEBSOCKET_AUDIO_BATCH_e = RAW_COMPLETE_AUDIO_TEST.shape[0] """

for i, pred in enumerate(tab):
    print(
        f"{pred['label']} - {pred['confidence']}% | {pred['start']:.2f}s - {pred['end']:.2f}s"
    )


outputaudio = []

""" 
D3 - 57% | 0.08s - 0.30s
G3 - 83% | 0.30s - 0.57s
A3 - 58% | 0.57s - 0.74s
C4 - 81% | 0.74s - 1.09s
A3 - 79% | 1.09s - 1.30s
A3 - 43% | 1.30s - 1.49s
C4 - 57% | 1.49s - 1.70s
D#5 - 12% | 1.70s - 1.81s
E4 - 83% | 1.81s - 2.07s
G4 - 96% | 2.07s - 2.52s
C4 - 88% | 2.52s - 2.96s
D4 - 75% | 2.50s - 2.93s
D4 - 68% | 2.93s - 3.14s
"""
for i, pred in enumerate(tab):
    found = False
    for root, dirs, _ in os.walk(os.path.join(ds_path)):
        for dir in dirs:
            if dir == pred["label"]:
                for file in os.listdir(os.path.join(root, dir)):
                    if file.endswith("wav"):
                        filepath = os.path.join(root, dir, file)
                        audio, sr = load(filepath)
                        # based on the start and the end, fill the outputaudio with the empty audio until the start, then with the audio of the file, then with the empty audio until the end

                        outputaudio.extend(audio)
                        # outputaudio.extend(np.zeros(int(0.02 * sr)))

                        found = True
                        break
                if found:
                    break
        if found:
            break

outputaudio = np.array(outputaudio)
sf.write("output.wav", outputaudio, sr)
