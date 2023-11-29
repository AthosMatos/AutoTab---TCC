import tensorflow as tf
from dataset_utils import index_directory
from Helpers.consts.paths import training_ds_path
from Helpers.DataPreparation.audioUtils import loadAndPrepare, load, Prepare
import numpy as np
import os
from keras.utils import to_categorical


# Function to load and process audio file
def load_audio(path, index, label):
    file_path = path.numpy().decode("utf-8")
    index = int(index.numpy())
    label = label.numpy()
    audio, sr = load(file_path)

    printStep = int(len(file_paths) * 0.05)

    if index % printStep == 0 or index == len(file_paths) - 1 or index == 0:
        os.system("cls")
        porcentaje = (index * 100) / len(file_paths)
        print(f"Loading...")
        print(f"{porcentaje:.2f}%")

    return audio, sr, index, label


def process_audio(waveform, sr, index, label):
    waveform = waveform.numpy()
    sr = sr.numpy()
    spec = Prepare(waveform, sr)
    label = label.numpy()
    labels = []
    if label == class_names.index("AMajor"):
        labels.append(to_categorical(class_names.index("A2"), len(class_names)))
        labels.append(to_categorical(class_names.index("E3"), len(class_names)))
        labels.append(to_categorical(class_names.index("A3"), len(class_names)))
        labels.append(to_categorical(class_names.index("C#4"), len(class_names)))
        labels.append(to_categorical(class_names.index("E4"), len(class_names)))
    elif label == class_names.index("AMinor"):
        labels.append(to_categorical(class_names.index("A2"), len(class_names)))
        labels.append(to_categorical(class_names.index("E3"), len(class_names)))
        labels.append(to_categorical(class_names.index("A3"), len(class_names)))
        labels.append(to_categorical(class_names.index("C4"), len(class_names)))
        labels.append(to_categorical(class_names.index("E4"), len(class_names)))
    elif label == class_names.index("BMajor"):
        labels.append(to_categorical(class_names.index("B2"), len(class_names)))
        labels.append(to_categorical(class_names.index("F#3"), len(class_names)))
        labels.append(to_categorical(class_names.index("B3"), len(class_names)))
        labels.append(to_categorical(class_names.index("D#4"), len(class_names)))
        labels.append(to_categorical(class_names.index("F#4"), len(class_names)))
    elif label == class_names.index("CMajor"):
        labels.append(to_categorical(class_names.index("C2"), len(class_names)))
        labels.append(to_categorical(class_names.index("G3"), len(class_names)))
        labels.append(to_categorical(class_names.index("C3"), len(class_names)))
        labels.append(to_categorical(class_names.index("E4"), len(class_names)))
        labels.append(to_categorical(class_names.index("G4"), len(class_names)))
    elif label == class_names.index("DMajor"):
        labels.append(to_categorical(class_names.index("D3"), len(class_names)))
        labels.append(to_categorical(class_names.index("A3"), len(class_names)))
        labels.append(to_categorical(class_names.index("D4"), len(class_names)))
        labels.append(to_categorical(class_names.index("F#4"), len(class_names)))
    elif label == class_names.index("DMinor"):
        labels.append(to_categorical(class_names.index("D3"), len(class_names)))
        labels.append(to_categorical(class_names.index("A3"), len(class_names)))
        labels.append(to_categorical(class_names.index("D4"), len(class_names)))
        labels.append(to_categorical(class_names.index("F4"), len(class_names)))
    elif label == class_names.index("EMajor"):
        labels.append(to_categorical(class_names.index("E2"), len(class_names)))
        labels.append(to_categorical(class_names.index("B2"), len(class_names)))
        labels.append(to_categorical(class_names.index("E3"), len(class_names)))
        labels.append(to_categorical(class_names.index("G#3"), len(class_names)))
        labels.append(to_categorical(class_names.index("B3"), len(class_names)))
        labels.append(to_categorical(class_names.index("E4"), len(class_names)))
    elif label == class_names.index("EMinor"):
        labels.append(to_categorical(class_names.index("E2"), len(class_names)))
        labels.append(to_categorical(class_names.index("B2"), len(class_names)))
        labels.append(to_categorical(class_names.index("E3"), len(class_names)))
        labels.append(to_categorical(class_names.index("G3"), len(class_names)))
        labels.append(to_categorical(class_names.index("B3"), len(class_names)))
        labels.append(to_categorical(class_names.index("E4"), len(class_names)))
    elif label == class_names.index("FMajor"):
        labels.append(to_categorical(class_names.index("F2"), len(class_names)))
        labels.append(to_categorical(class_names.index("C2"), len(class_names)))
        labels.append(to_categorical(class_names.index("F3"), len(class_names)))
        labels.append(to_categorical(class_names.index("A3"), len(class_names)))
        labels.append(to_categorical(class_names.index("C4"), len(class_names)))
        labels.append(to_categorical(class_names.index("F4"), len(class_names)))
    elif label == class_names.index("GMajor"):
        labels.append(to_categorical(class_names.index("G2"), len(class_names)))
        labels.append(to_categorical(class_names.index("D2"), len(class_names)))
        labels.append(to_categorical(class_names.index("G3"), len(class_names)))
        labels.append(to_categorical(class_names.index("B3"), len(class_names)))
        labels.append(to_categorical(class_names.index("D4"), len(class_names)))
        labels.append(to_categorical(class_names.index("G4"), len(class_names)))
    else:
        labels.append(to_categorical(label, len(class_names)))
    if len(labels) < 6:
        for _ in range(6 - len(labels)):
            labels.append(to_categorical(class_names.index("noise"), len(class_names)))
    labels = np.array(labels)
    printStep = int(len(file_paths) * 0.05)

    if index % printStep == 0 or index == len(file_paths) - 1 or index == 0:
        os.system("cls")
        porcentaje = (index * 100) / len(file_paths)
        print(f"Preprocessing...")
        print(f"{porcentaje:.2f}%")
    return (spec, labels)


file_paths, labels, class_names = index_directory(
    training_ds_path,
    "inferred",
    formats=".wav",
    class_names=None,
    shuffle=True,
    seed=0,
    follow_links=False,
)

# print(class_names)


paths = []
indexes = []
for i, path in enumerate(file_paths):
    paths.append(path)
    indexes.append(i)

# Create a dataset from file paths
path_ds = tf.data.Dataset.from_tensor_slices(paths)
index_ds = tf.data.Dataset.from_tensor_slices(indexes)
labels_ds = tf.data.Dataset.from_tensor_slices(labels)

dataset = tf.data.Dataset.zip((path_ds, index_ds, labels_ds))


# Define a function to apply to each element in the dataset
def process_path(path, index, label):
    return tf.py_function(
        load_audio, [path, index, label], [tf.float32, tf.int32, tf.float32, tf.int32]
    )


spectrogram_dataset = dataset.map(
    process_path,
    num_parallel_calls=tf.data.AUTOTUNE,
)


def process_spec(audio, sr, index, label):
    res = tf.py_function(
        process_audio, [audio, sr, index, label], [tf.float32, tf.float32]
    )
    # define shape of output
    res[0].set_shape([84, 216, 1])
    res[1].set_shape([6, len(class_names)])
    return res


def make_spec_ds(ds):
    return ds.map(
        map_func=process_spec,
        num_parallel_calls=tf.data.AUTOTUNE,
    )


train_spectrogram_ds = make_spec_ds(spectrogram_dataset)

train_spectrogram_ds = (
    train_spectrogram_ds.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
)

x_train = []
y_train = []
for s, l in list(train_spectrogram_ds.as_numpy_iterator()):
    x_train.append(s)
    y_train.append(l)

x_train = np.array(x_train)
y_train = np.array(y_train)
print(x_train.shape)
print(y_train.shape)


""" 
import tensorflow as tf
from dataset_utils import index_directory
from Helpers.consts.paths import training_ds_path
from Helpers.DataPreparation.audioUtils import loadAndPrepare, load, Prepare
import numpy as np
import os
from keras.utils import to_categorical


# Function to load and process audio file
def load_audio(path, index):
    file_path = path.numpy().decode("utf-8")
    index = int(index.numpy())
    audio, sr = load(file_path)

    printStep = int(len(file_paths) * 0.05)

    if index % printStep == 0 or index == len(file_paths) - 1 or index == 0:
        os.system("cls")
        porcentaje = (index * 100) / len(file_paths)
        print(f"Loading...")
        print(f"{porcentaje:.2f}%")

    return audio, sr, index


def process_audio(waveform, sr, index):
    waveform = waveform.numpy()
    sr = sr.numpy()
    spec = Prepare(waveform, sr)

    printStep = int(len(file_paths) * 0.05)

    if index % printStep == 0 or index == len(file_paths) - 1 or index == 0:
        os.system("cls")
        porcentaje = (index * 100) / len(file_paths)
        print(f"Preprocessing...")
        print(f"{porcentaje:.2f}%")
    return spec


file_paths, labels, class_names = index_directory(
    training_ds_path,
    "inferred",
    formats=".wav",
    class_names=None,
    shuffle=True,
    seed=0,
    follow_links=False,
)

print(labels.shape)

paths = []
indexes = []
for i, path in enumerate(file_paths):
    paths.append(path)
    indexes.append(i)

# Create a dataset from file paths
path_ds = tf.data.Dataset.from_tensor_slices(paths)
labels_ds = tf.data.Dataset.from_tensor_slices(labels)
index_ds = tf.data.Dataset.from_tensor_slices(indexes)

dataset = tf.data.Dataset.zip((path_ds, index_ds))


# Define a function to apply to each element in the dataset
def process_path(path, index):
    return tf.py_function(load_audio, [path, index], [tf.float32, tf.int32, tf.float32])


spectrogram_dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)


def process_spec(audio, sr, index):
    return tf.py_function(process_audio, [audio, sr, index], [tf.float32])


def make_spec_ds(ds):
    return ds.map(
        map_func=process_spec,
        num_parallel_calls=tf.data.AUTOTUNE,
    )


train_spectrogram_ds = make_spec_ds(spectrogram_dataset)

for example_spectrograms in train_spectrogram_ds.take(1):
    print(example_spectrograms[0].shape)


exit()

x_train = []
y_train = []


# Iterate over the dataset to save spectrograms
for i, spectrogram in enumerate(spectrogram_dataset):
    x_train.append(spectrogram)
    # to_categorical
    y_train.append(to_categorical(labels[i], len(class_names)))

x_train = np.array(x_train)
y_train = np.array(y_train)

print(x_train.shape)
print(y_train.shape)


"""
