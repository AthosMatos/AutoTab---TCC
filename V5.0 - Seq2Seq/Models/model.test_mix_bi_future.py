from numpy import array, argmax, zeros, random
from keras.models import Model
from keras.layers import (
    Input,
    LSTM,
    Dense,
    Bidirectional,
    Concatenate,
    Attention,
    concatenate,
    Conv2D,
    Flatten,
    MaxPooling2D,
)
from keras.losses import (
    MeanSquaredError,
    CategoricalCrossentropy,
)
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from random import randint
from keras.utils import plot_model
import tensorflow as tf


# generate a sequence of random integers
def generate_sequence(length, empty=False):
    if empty:
        return zeros((length, 2)).round(2).tolist()
    # random float between 0 and 5
    return random.uniform(0, 5, size=(length, 2)).round(2).tolist()


def generate_sequence_hot(length, n_unique, empty=False):
    if empty:
        return [0 for _ in range(length)]
    return [randint(1, n_unique - 1) for _ in range(length)]


# prepare data for the LSTM
def gen_dataset(n_in, n_out, n_features, n_out_gen, n_samples):
    X_TIMES, PREVS_TIMES, y_TIMES = list(), list(), list()
    X_NOTES, PREVS_NOTES, y_NOTES = list(), list(), list()
    X_SPEC = list()

    # random int between 1 and n_out
    if n_out_gen is None:
        n_out_gen = randint(2, n_out)

    for _ in range(n_samples):
        # n_out_gen = randint(2, n_out)

        source_times = generate_sequence(n_in)
        source_notes = generate_sequence_hot(n_in, n_features)

        """ print(f"source_times: {source_times}")
        print(f"source_notes: {source_notes}") """

        t_src = source_times[:n_out_gen]
        t_src.reverse()

        nt_src = source_notes[:n_out_gen]
        nt_src.reverse()

        target_times = pad_sequences(
            [t_src], maxlen=n_out, padding="pre", dtype="float"
        )[0]

        shift = 1
        target_in_times = pad_sequences(
            [t_src[:-shift]],
            maxlen=n_out,
            padding="pre",
            dtype="float",
        )[0]

        X_TIMES.append(source_times)
        PREVS_TIMES.append(target_in_times)
        y_TIMES.append(target_times)

        target_notes = pad_sequences(
            [nt_src], maxlen=n_out, padding="pre", dtype="float"
        )[0]
        target_in_notes = pad_sequences(
            [nt_src[:-shift]],
            maxlen=n_out,
            padding="pre",
            dtype="float",
        )[0]

        X_NOTES.append(to_categorical(source_notes, num_classes=n_features))
        PREVS_NOTES.append(to_categorical(target_in_notes, num_classes=n_features))
        y_NOTES.append(to_categorical(target_notes, num_classes=n_features))

        # generate spectogram fo 32x32 with zeros
        X_SPEC.append(zeros((32, 32, 1)))

    X_TIMES, PREVS_TIMES, y_TIMES = array(X_TIMES), array(PREVS_TIMES), array(y_TIMES)
    X_NOTES, PREVS_NOTES, y_NOTES = array(X_NOTES), array(PREVS_NOTES), array(y_NOTES)
    X_SPEC = array(X_SPEC)
    """ print(f"X_TIMES: {X_TIMES.shape}")
    print(f"PREVS_TIMES: {PREVS_TIMES.shape}")
    print(f"y_TIMES: {y_TIMES.shape}")

    print(f"X_NOTES: {X_NOTES.shape}")
    print(f"PREVS_NOTES: {PREVS_NOTES.shape}")
    print(f"y_NOTES: {y_NOTES.shape}") """

    return (X_TIMES, PREVS_TIMES, y_TIMES), (X_NOTES, PREVS_NOTES, y_NOTES), X_SPEC


# decode a one hot encoded string
def one_hot_decode_seq(encoded_seq):
    return [argmax(vector) for vector in encoded_seq]


def one_hot_decode(encoded):
    return argmax(encoded)


# returns train, inference_encoder and inference_decoder models
def define_models(n_input_output, n_steps_in, n_features, n_units):
    max_prev_events = n_steps_in
    n_times_IN_OUT = n_input_output
    n_classes = n_features

    latent_dim = n_units

    width_n_height = 32

    encoder_inputs_spectogram = Input(
        shape=(width_n_height, width_n_height, 1), name="encoder_inputs_spectogram"
    )

    conv = Conv2D(
        32, (3, 3), activation="relu", padding="same", name="encoder_conv2d_1"
    )(encoder_inputs_spectogram)
    maxPool = MaxPooling2D(pool_size=(2, 2), name="encoder_maxpool_1")(conv)

    conv = Conv2D(
        64, (3, 3), activation="relu", padding="same", name="encoder_conv2d_2"
    )(maxPool)
    maxPool = MaxPooling2D(pool_size=(2, 2), name="encoder_maxpool_2")(conv)
    flat = Flatten(name="encoder_flatten")(maxPool)

    spec_out = Dense(128, activation="relu", name="spec_out")(flat)

    # spectogram_model = Model(encoder_inputs_spectogram, spec_out, name="spectogram")

    """ plot_model(
        spectogram_model,
        to_file="FUTURE_spectogram_model_mix.png",
        show_layer_names=True,
        show_shapes=True,
    ) """

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

    state_h_encoder = concatenate(
        [state_h_encoder_f, state_h_encoder_b, spec_out], axis=-1
    )
    state_c_encoder = concatenate(
        [state_c_encoder_f, state_c_encoder_b, spec_out], axis=-1
    )

    print(state_h_encoder.shape)
    print(state_c_encoder.shape)

    encoder_states = [state_h_encoder, state_c_encoder]

    # Define inference encoder
    encoder_model = Model(
        [encoder_inputs_spectogram, encoder_inputs_times, encoder_inputs_notes],
        encoder_states,
        name="encoder",
    )

    # Plot model

    """ plot_model(
        encoder_model,
        to_file="FUTURE_encoder_model_mix.png",
        show_layer_names=True,
        show_shapes=True,
    ) """

    # Define training decoder
    decoder_inputs_times = Input(
        shape=(None, n_times_IN_OUT), name="decoder_inputs_times"
    )
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

    """ plot_model(
        model,
        to_file="FUTURE_model_mix.png",
        show_layer_names=True,
        show_shapes=True,
    ) """

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

    """ plot_model(
        decoder_model,
        to_file="FUTURE_decoder_model_mix.png",
        show_layer_names=True,
        show_shapes=True,
    ) """
    model.summary()

    return model, encoder_model, decoder_model


# generate target given source sequence
def predict_sequence(infenc, infdec, source, n_steps):
    X_TIMES = source[0]
    X_NOTES = source[1]
    X_SPEC = source[2]

    """ print(f"X_TIMES: {X_TIMES.shape}")
    print(f"X_NOTES: {X_NOTES.shape}") """

    # encode
    states = infenc.predict([X_SPEC, X_TIMES, X_NOTES], verbose=0)

    # print(f"states: {len(states)}")

    # start of sequence input
    target_seq_times = array([generate_sequence(1, empty=True)])
    target_seq_notes = array(
        [
            to_categorical(
                generate_sequence_hot(1, n_classes, empty=True), num_classes=n_classes
            )
        ]
    )

    """ print(f"target_seq_times: {target_seq_times.shape}")
    print(f"target_seq_notes: {target_seq_notes.shape}") """

    # print(f"target_seq: {target_seq.shape}")
    # collect predictions
    output_times = list()
    output_notes = list()

    for _ in range(n_steps):
        # predict next char
        # infdec.predict(
        times_out, notes_out, h, c = infdec.predict(
            [target_seq_times, target_seq_notes] + states, verbose=0
        )
        """ 
        # store prediction
        print(f"times_out: {times_out.shape}")
        print(f"notes_out: {notes_out.shape}") 
        """

        output_times.append(times_out[0, 0, :].round(2))
        output_notes.append(notes_out[0, 0, :])

        # update state
        states = [h, c]
        # update target sequence
        target_seq_times = times_out
        target_seq_notes = notes_out

        """ print(f"out times: ", end=" ")
        for out in output_times:
            print(out, end=" ")
        print()
        print(f"out notes: ", end=" ")
        for out in output_notes:
            print(one_hot_decode(out), end=" ")

        print() """
        # print(one_hot_decode(output))
    return array(output_times), array(output_notes)


# configure problem
n_classes = 50 + 1
n_steps_in = 20
n_steps_out = 10
n_out_gen = 8

# define model
train_model, encoder, decoder = define_models(2, n_steps_in, n_classes, 128)

train_model.compile(
    optimizer=Adam(),  # "rmsprop",
    loss={
        "decoder_outputs_times": MeanSquaredError(),
        "decoder_outputs_notes": CategoricalCrossentropy(),
    },
    metrics=["accuracy"],
)


# n_steps_out_gen = 8
# generate training dataset
TIMES, NOTES, SPECS = gen_dataset(n_steps_in, n_steps_out, n_classes, n_out_gen, 10000)

# (X_TIMES, PREVS_TIMES, y_TIMES)

X_TIMES, PREVS_TIMES, y_TIMES = TIMES
X_NOTES, PREVS_NOTES, y_NOTES = NOTES

""" for i in range(10):
    explaination = (
        "The X values are the times and the notes previously predicted"
        + "(historic) and the y values are the times and notes to be predicted (future).\n"
        + "The Prevs (or X2) values are like a road turn decider: \n"
        + ""
    )

    print(f"X: ", end=" ")
    for j, x in enumerate(X_TIMES[i]):
        print(f"{x}({one_hot_decode(X_NOTES[i][j])})", end=" ")
    print()
    print(f"Prevs: ", end=" ")
    for j, prev in enumerate(PREVS_TIMES[i]):
        print(f"{prev}({one_hot_decode(PREVS_NOTES[i][j])})", end=" ")
    print()
    print(f"y: ", end=" ")
    for j, y_ in enumerate(y_TIMES[i]):
        print(f"{y_}({one_hot_decode(y_NOTES[i][j])})", end=" ")
    print()
    print() """


# train model
train_model.fit(
    [SPECS, X_TIMES, X_NOTES, PREVS_TIMES, PREVS_NOTES],
    [y_TIMES, y_NOTES],
    epochs=10,
)

# spot check some examples
for _ in range(10):
    TIMES, NOTES, X_SPEC = gen_dataset(n_steps_in, n_steps_out, n_classes, n_out_gen, 1)

    X_TIMES, PREVS_TIMES, y_TIMES = TIMES
    X_NOTES, PREVS_NOTES, y_NOTES = NOTES
    predict = predict_sequence(
        encoder, decoder, [X_TIMES, X_NOTES, X_SPEC], n_steps_out
    )

    print(f"X: ", end=" ")
    for j, x in enumerate(X_TIMES[0]):
        print(f"{x}({one_hot_decode(X_NOTES[0][j])})", end=" ")
    print()
    """ print(f"Prevs: ", end=" ")
    for j, prev in enumerate(PREVS_TIMES[0]):
        print(f"{prev}({one_hot_decode(PREVS_NOTES[0][j])})", end=" ")
    print() """
    print(f"y: ", end=" ")
    for j, y_ in enumerate(y_TIMES[0]):
        print(f"{y_}({one_hot_decode(y_NOTES[0][j])})", end=" ")
    print()
    print(f"p: ", end=" ")
    for j, y_ in enumerate(predict[0]):
        print(f"{y_}({one_hot_decode(predict[1][j])})", end=" ")

    print()
    print()
