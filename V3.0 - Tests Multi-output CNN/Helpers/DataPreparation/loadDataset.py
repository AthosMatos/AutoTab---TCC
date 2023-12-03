import os
import numpy as np
from keras.utils import to_categorical
from Helpers.DataPreparation.audioUtils import loadAndPrepare
from Helpers.consts.paths import training_ds_path
import os


def loadDS(maxFiles=None, split=(0.6, 0.3, 0.1)):
    audio_inputs = []
    notes_outputs = []
    labels = []
    stop = False

    audio_train = []
    labls_train = []
    audio_val = []
    labls_val = []
    audio_test = []
    labls_test = []

    if maxFiles != None:
        amount_of_files = maxFiles
    else:
        amount_of_files = 0
        for root, dirs, files in os.walk(training_ds_path):
            for dir in dirs:
                for file in os.listdir(os.path.join(root, dir)):
                    if file.endswith("wav"):
                        amount_of_files += 1

    """ labels_ignore = ["Major", "Minor"] """

    for root, dirs, _ in os.walk(training_ds_path):
        # use folders as classes notes_outputs
        for dir in dirs:
            # if dir has substring in labels_ignore, ignore it
            """if any(label in dir for label in labels_ignore):
            continue"""
            FILES = os.listdir(os.path.join(root, dir))
            MAX_TRAIN = int(len(FILES) * split[0])
            MAX_VAL = int(len(FILES) * split[1])
            MAX_TEST = int(len(FILES) * split[2])

            t = 0
            v = 0

            for file in FILES:
                if file.endswith("wav"):
                    wav_file_path = os.path.join(root, dir, file)
                    S, _ = loadAndPrepare(wav_file_path, (None, None))

                    if t < MAX_TRAIN:
                        audio_train.append(S)
                        labls_train.append(dir)
                        t += 1
                    elif v < MAX_VAL:
                        audio_val.append(S)
                        labls_val.append(dir)
                        v += 1
                    else:
                        audio_test.append(S)
                        labls_test.append(dir)

                    audio_inputs.append(S)
                    notes_outputs.append(dir)

                    os.system("cls")
                    # print folder
                    print(f"Folder: {dir}")
                    """  # print filename
                    print(f"File: {file}") """
                    # print completion porcentage
                    print(
                        f"Loading training dataset: {round((len(audio_inputs) / amount_of_files) * 100, 2)}%"
                    )

                    if maxFiles != None and len(audio_inputs) >= maxFiles:
                        stop = True
                        break
            if stop:
                break
        if stop:
            break

    def prepareOutput(output, labels):
        output = np.array(output)
        category_mapping = {category: i for i, category in enumerate(labels)}
        output = to_categorical([category_mapping[category] for category in output])
        return output

    # save the x_train and y_train as numpy arrays
    audio_inputs = np.array(audio_inputs)
    labels = np.unique(notes_outputs)
    notes_outputs = prepareOutput(notes_outputs, labels)
    audio_train = np.array(audio_train)
    labls_train = prepareOutput(labls_train, labels)
    audio_val = np.array(audio_val)
    labls_val = prepareOutput(labls_val, labels)
    audio_test = np.array(audio_test)
    labls_test = prepareOutput(labls_test, labels)

    os.system("cls")
    print(f"Found {len(audio_inputs)} files")
    print(f"Training dataset: {(audio_train.shape)} | {labls_train.shape}")
    print(f"Validation dataset: {(audio_val.shape)} | {labls_val.shape}")
    print(f"Test dataset: {(audio_test.shape)} | {labls_test.shape}")
    print(f"With {len(labels)} labels")

    print(f"Spectogram shape: {audio_inputs[0].shape}")
    print("labels: ", labels)

    return (
        ((audio_train, labls_train), (audio_val, labls_val), (audio_test, labls_test)),
        (audio_inputs, notes_outputs),
        labels,
    )
