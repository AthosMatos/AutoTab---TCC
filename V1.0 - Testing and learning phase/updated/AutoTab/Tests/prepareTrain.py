import os
import json
import numpy as np
from scipy.io import wavfile
from keras.utils import to_categorical
from math import log2, pow
import librosa
from sklearn.preprocessing import StandardScaler

nftt = 1024
hop_length = nftt // 4
all_notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def pitch_to_note(freq):
    A4 = 440
    C0 = A4*pow(2, -4.75)

    h = round(12*log2(freq/C0))
    octave = h // 12
    n = h % 12
    return all_notes[n] + str(octave)


# preprocessing function
def preprocess(batch, sr):
    S = librosa.feature.melspectrogram(
        y=batch, sr=sr, n_fft=nftt, hop_length=hop_length, n_mels=256)
    S = librosa.power_to_db(S, ref=np.max)

    # Reshape the data to 2D for standardization
    X_2d = S.reshape(-1, S.shape[-1])

    # Standardize the data
    scaler = StandardScaler()
    X_2d_scaled = scaler.fit_transform(X_2d)

    # Reshape the data back to its original shape
    X_scaled = X_2d_scaled.reshape(S.shape)

    S = librosa.util.normalize(X_scaled, norm=1, fill=True)
    return S


# Function to load wav file and create batches
def process_wav_file(wav_file_path):
    sample_rate, data = wavfile.read(wav_file_path + '.wav')
    data = data.astype(np.float16)
    batch_size = int(sample_rate * 0.12)  # 0.12 seconds of data
    batches = []

    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        if len(batch) < batch_size:
            # Pad with zeros
            batch = np.pad(batch, (0, batch_size - len(batch)))
        preprocessed_batch = preprocess(batch, sample_rate)
        batches.append(preprocessed_batch)
    batches = np.array(batches)

    return batches

# Main function to process the dataset


def process_dataset(train_dataset_path):
    all_batches = []
    all_pitches = []

    data = json.load(open(os.path.join(train_dataset_path, 'examples.json')))
    for key in data.keys():
        wav_file_path = os.path.join(train_dataset_path + '/audio', key)
        print('Processing: ', wav_file_path)

        # Get the batches for this wav file
        batches = process_wav_file(wav_file_path)
        # Get the note for this wav file
        pitch = data[key]['pitch']
        # Pitch to midi
        pitch = pitch_to_note(float(pitch))

        all_batches.append(batches)
        all_pitches.append(pitch)

    all_batches = np.array(all_batches)
    all_pitches = np.array(all_pitches)
    unique_pitches = np.unique(all_pitches)
    print('batches anmount: ', all_batches.shape[0])
    print('Unique pitches: ', unique_pitches.shape[0])

    # Assign numerical values to categories
    category_mapping = {category: i for i,
                        category in enumerate(unique_pitches)}

    # Convert string array to categorical array
    y_train = to_categorical([category_mapping[category]
                             for category in all_pitches])

    np.save('x_train.npy', all_batches)
    np.save('y_train.npy', y_train)


# Run the script
process_dataset('Datasets/Nsynth/nsynth-train')
