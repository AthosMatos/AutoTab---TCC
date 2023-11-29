import numpy as np
from Helpers.DataPreparation.audioUtils import load, rmsNorm, Prepare
import os
import librosa
from keras.models import load_model
from Helpers.consts.labels import labels
import soundfile as sf
from Helpers.consts.paths import ds_path

model = load_model("model-[32,32]-tanh-[-1,1]-out6.keras")
# model = load_model("NotesNeuralNet.h5")

# is like the wesocket channel awaiting for the music to come
""" CHANNELTESTPATH = os.path.dirname(__file__) + "/dataset/musics/my bron-yr-aur.mp3"
RAW_COMPLETE_AUDIO_TEST, SR = load(CHANNELTESTPATH, (0, 7)) """

""" # is like the wesocket channel awaiting for the music to come
CHANNELTESTPATH = os.path.dirname(__file__) + "/dataset/musics/beach house - clean.wav"
RAW_COMPLETE_AUDIO_TEST, SR = load(CHANNELTESTPATH, (0, 20)) """

CHANNELTESTPATH = os.path.dirname(__file__) + "/dataset/musics/TEST SONG.wav"
RAW_COMPLETE_AUDIO_TEST, SR = load(CHANNELTESTPATH)
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
    audio_cqt_spec = np.abs(
        librosa.cqt(
            y=RMS_NORM_RAW_AUDIO_BATCH,
            sr=SR,
        )
    )
    onset_strenght = librosa.onset.onset_strength(S=audio_cqt_spec, sr=SR)
    onset_times = librosa.times_like(onset_strenght, sr=SR)
    """ onset_raw = librosa.onset.onset_detect(
        onset_envelope=onset_strenght, backtrack=False, sr=SR
    ) """

    onsets = []
    for i in range(onset_times.shape[0] - 1):
        onset_start = onset_times[i]
        onsets_end = onset_times[i + 1]
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
        sum_of_confidence = 0
        notes_preds = []
        for i, pred in enumerate(y_pred[0]):
            confidence = np.max(pred) * 100
            sum_of_confidence += confidence
            "save the notes"
            notes_preds.append(labels[np.argmax(pred)])
        "get the sum of all confiedences"
        sum_of_confidence = (sum_of_confidence * 100) / 600
        print(f"confidence: {sum_of_confidence:.2f}%")
        for note in notes_preds:
            print(f"|| {note} ||")

        print(
            f"{FIX_TIMING_ONSET_TIMES[i][0]:.2f}s - {FIX_TIMING_ONSET_TIMES[i][1]:.2f}s",
        )
        # if labels[best_pred] != "noise":
        # preds.append((labels[best_pred], int(confidence)))
        preds.append(
            {
                "labels": notes_preds,
                "confidence": int(sum_of_confidence),
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
        f"{pred['labels']} - {pred['confidence']}% | {pred['start']:.2f}s - {pred['end']:.2f}s"
    )

exit()

""" 
E3 - 54% | 1.36s - 1.61s
C4 - 41% | 1.61s - 1.90s
D#3 - 35% | 1.90s - 2.14s
C#3 - 35% | 2.37s - 2.58s
C3 - 50% | 2.58s - 2.82s
D#3 - 82% | 2.53s - 2.70s
C3 - 57% | 2.79s - 3.02s
CMajor - 57% | 3.02s - 3.27s
F2 - 27% | 3.29s - 3.51s
D#3 - 39% | 3.51s - 3.75s
D#3 - 57% | 3.75s - 3.97s
B2 - 25% | 3.97s - 4.23s
B2 - 14% | 4.23s - 4.49s
D#3 - 71% | 4.49s - 4.66s
C3 - 51% | 4.66s - 4.90s
CMajor - 30% | 4.90s - 5.16s
C#3 - 28% | 5.16s - 5.37s
D#3 - 37% | 5.10s - 5.22s
E2 - 19% | 5.34s - 5.58s
C#3 - 33% | 5.58s - 5.73s
"""

outputaudio = []

# based on this pront, output a audio file with the sounds of the log
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
