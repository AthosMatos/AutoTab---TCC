from keras.models import Model
from keras.layers import (
    Input,
    LSTM,
    Dense,
    Bidirectional,
    concatenate,
    Conv2D,
    Flatten,
    MaxPooling2D,
)
from keras.utils import plot_model


max_prev_events = 100
n_times_IN_OUT = 2
n_classes = 51

latent_dim = 128

width_n_height = 32

encoder_inputs_spectogram = Input(
    shape=(width_n_height, width_n_height, 1), name="encoder_inputs_spectogram"
)

conv = Conv2D(32, (3, 3), activation="relu", padding="same", name="encoder_conv2d_1")(
    encoder_inputs_spectogram
)
maxPool = MaxPooling2D(pool_size=(2, 2), name="encoder_maxpool_1")(conv)

conv = Conv2D(64, (3, 3), activation="relu", padding="same", name="encoder_conv2d_2")(
    maxPool
)
maxPool = MaxPooling2D(pool_size=(2, 2), name="encoder_maxpool_2")(conv)
flat = Flatten(name="encoder_flatten")(maxPool)

spec_out = Dense(128, activation="relu", name="spec_out")(flat)


spectogram_model = Model(encoder_inputs_spectogram, spec_out, name="spectogram")


plot_model(
    spectogram_model,
    to_file="FUTURE_spectogram_model_mix.png",
    show_layer_names=True,
    show_shapes=True,
)

# Define training encode
encoder_inputs_times = Input(
    shape=(max_prev_events, n_times_IN_OUT), name="encoder_inputs_times"
)
encoder_inputs_notes = Input(
    shape=(max_prev_events, n_classes), name="encoder_inputs_notes"
)

encoder_inputs_merged = concatenate(
    [encoder_inputs_times, encoder_inputs_notes], axis=-1
)

encoder = Bidirectional(LSTM(latent_dim, return_state=True, name="encoder_lstm"))
(
    _,
    state_h_encoder_f,
    state_c_encoder_f,
    state_h_encoder_b,
    state_c_encoder_b,
) = encoder(encoder_inputs_merged)

state_h_encoder = concatenate([state_h_encoder_f, state_h_encoder_b, spec_out], axis=-1)
state_c_encoder = concatenate([state_c_encoder_f, state_c_encoder_b, spec_out], axis=-1)

print(state_h_encoder.shape)
print(state_c_encoder.shape)

encoder_states = [state_h_encoder, state_c_encoder]

# Define inference encoder
encoder_model = Model(
    [encoder_inputs_times, encoder_inputs_notes, encoder_inputs_spectogram],
    encoder_states,
    name="encoder",
)

# Plot model

plot_model(
    encoder_model,
    to_file="FUTURE_encoder_model_mix.png",
    show_layer_names=True,
    show_shapes=True,
)

# Define training decoder
decoder_inputs_times = Input(shape=(None, n_times_IN_OUT), name="decoder_inputs_times")
decoder_inputs_notes = Input(shape=(None, n_classes), name="decoder_inputs_notes")

decoder_inputs_merged = concatenate(
    [decoder_inputs_times, decoder_inputs_notes], axis=-1
)

decoder = LSTM(
    (latent_dim * 2) + latent_dim,
    return_sequences=True,
    return_state=True,
    name="decoder_lstm",
)

decoder_outputs, _, _ = decoder(decoder_inputs_merged, initial_state=encoder_states)


decoder_dense_times = Dense(
    n_times_IN_OUT, activation="linear", name="decoder_outputs_times"
)
decoder_outputs_times = decoder_dense_times(decoder_outputs)


decoder_dense_notes = Dense(
    n_classes, activation="softmax", name="decoder_outputs_notes"
)
decoder_outputs_notes = decoder_dense_notes(decoder_outputs)

model_inputs = [
    encoder_inputs_spectogram,
    encoder_inputs_times,
    encoder_inputs_notes,
    decoder_inputs_times,
    decoder_inputs_notes,
]
model_outputs = [decoder_outputs_times, decoder_outputs_notes]

model = Model(model_inputs, model_outputs, name="model")

plot_model(
    model,
    to_file="FUTURE_model_mix.png",
    show_layer_names=True,
    show_shapes=True,
)

# Define inference decoder
decoder_state_input_h = Input(
    shape=((latent_dim * 2) + latent_dim,), name="decoder_state_input_h"
)
decoder_state_input_c = Input(
    shape=((latent_dim * 2) + latent_dim,), name="decoder_state_input_c"
)

decoder_outputs, state_h, state_c = decoder(
    decoder_inputs_merged,
    initial_state=[decoder_state_input_h, decoder_state_input_c],
)

decoder_outputs_times = decoder_dense_times(decoder_outputs)
decoder_outputs_notes = decoder_dense_notes(decoder_outputs)

decoder_model = Model(
    [decoder_inputs_times, decoder_inputs_notes]
    + [decoder_state_input_h, decoder_state_input_c],
    [decoder_outputs_times, decoder_outputs_notes] + [state_h, state_c],
    name="decoder",
)

plot_model(
    decoder_model,
    to_file="FUTURE_decoder_model_mix.png",
    show_layer_names=True,
    show_shapes=True,
)
model.summary()
