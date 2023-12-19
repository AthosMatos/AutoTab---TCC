import numpy as np
from keras.layers import (
    Conv2D,
    Dense,
    MaxPooling2D,
    Activation,
    Flatten,
    Dropout,
    Input,
    BatchNormalization,
    GlobalAveragePooling2D,
    LSTM,
    Concatenate,
)
from keras.losses import BinaryCrossentropy, BinaryFocalCrossentropy
from keras.callbacks import Callback
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Example target sequences (list of strings)
target_sequences = ["C2", "C3", "C4"]

# Tokenize target sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(target_sequences)

# Convert text to sequences of indices
y_train_sequences = tokenizer.texts_to_sequences(target_sequences)

# Pad sequences to a fixed length
max_sequence_length = max(len(seq) for seq in y_train_sequences)
y_train_padded = pad_sequences(
    y_train_sequences, maxlen=max_sequence_length, padding="post"
)

# Example of y_train_padded
print(y_train_padded)
exit()

train_ds, unique_labels = (
    np.load("/content/drive/MyDrive/AutoTAB/chords_np_cqt_44.1k/train_ds-6out.npz"),
    np.load(
        "/content/drive/MyDrive/AutoTAB/chords_np_cqt_44.1k/unique_labels-6out.npy"
    ),
)

train_x = train_ds["x"]
train_y = []
# fill train_y with all the values from train_ds["y"]
for i in range(len(unique_labels)):
    train_y.append(train_ds["y"][:, i])


# Assuming these values for illustration purposes
height = train_x.shape[1]
width = train_x.shape[2]
channels = train_x.shape[3]
latent_dim = 256
sequence_length = 10
input_dim = 50  # Dimensionality of each time step in the sequence
output_dim = 10000  # Vocabulary size for machine translation

# Define image input
image_inputs = Input(shape=(height, width, channels), name="image_input")

# CNN to extract features from images
conv1 = Conv2D(64, (3, 3), activation="relu", name="conv1")(image_inputs)
pool1 = MaxPooling2D(pool_size=(2, 2), name="pool1")(conv1)
flatten = Flatten(name="flatten")(pool1)

# Define sequence input
sequence_inputs = Input(shape=(sequence_length, input_dim), name="sequence_input")

# Combine image features and sequence inputs
combined_inputs = Concatenate(name="concatenated_inputs")([flatten, sequence_inputs])

# LSTM for sequence-to-sequence
encoder_lstm = LSTM(
    latent_dim, return_sequences=True, return_state=True, name="encoder_lstm"
)
encoder_outputs, state_h, state_c = encoder_lstm(combined_inputs)
encoder_states = [state_h, state_c]

# Define the decoder LSTM
decoder_lstm = LSTM(
    latent_dim, return_sequences=True, return_state=True, name="decoder_lstm"
)
decoder_outputs, _, _ = decoder_lstm(sequence_inputs, initial_state=encoder_states)

# Additional dense layer for output
decoder_dense = Dense(output_dim, activation="softmax", name="output_dense")
output = decoder_dense(decoder_outputs)

# Define the model
model = Model(inputs=[image_inputs, sequence_inputs], outputs=output)

# Compile the model
model.compile(
    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
)
