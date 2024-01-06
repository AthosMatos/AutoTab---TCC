from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.metrics import CategoricalAccuracy, MeanAbsoluteError

from keras.losses import categorical_crossentropy, mean_squared_error
from keras.layers import (
    LSTM,
    TimeDistributed,
    Input,
    Dense,
    RepeatVector,
    Conv2D,
    MaxPooling2D,
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dropout,
    Bidirectional,
    Activation,
    Reshape,
    BatchNormalization,
    Concatenate,
)
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint, EarlyStopping


# define mode

spec_width = 32
spec_height = 32
spec_channels = 1

TimesSteps = 14
latent_dim = 128

spec_input = Input(shape=(spec_width, spec_height, spec_channels))
spec_conv_layer = Conv2D(32, (3, 3), activation="relu", padding="same")(spec_input)
spec_conv_layer = MaxPooling2D((2, 2), padding="same")(spec_conv_layer)
spec_conv_layer = Conv2D(64, (3, 3), activation="relu", padding="same")(spec_conv_layer)
spec_conv_layer = MaxPooling2D((2, 2), padding="same")(spec_conv_layer)
spec_flat = Flatten()(spec_conv_layer)
spec_out = Dense(128, activation="relu")(spec_flat)

# spectogram_model = Model(Spectogram_input, out)

times_input = Input(shape=(TimesSteps, 1))
times_output = LSTM(latent_dim)(times_input)

notes_input = Input(shape=(TimesSteps, 1))
notes_output = LSTM(latent_dim)(notes_input)

times_notes_concat_out = Concatenate()([times_output, notes_output])

# temporal_model = Model([times_input, notes_input], times_notes_concat)

spec_temporal_concat = Concatenate()([spec_out, times_notes_concat_out])

spec_temporal_model = Model(
    [spec_input, times_input, notes_input], spec_temporal_concat
)

spec_temporal_model.summary()

# plot model
from keras.utils import plot_model

plot_model(
    spec_temporal_model,
    to_file="spec_temporal_model.png",
    show_shapes=True,
    show_layer_names=True,
    
)
