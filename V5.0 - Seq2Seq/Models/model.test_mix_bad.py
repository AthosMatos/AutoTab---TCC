from numpy import array, argmax, zeros, random
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Bidirectional, Concatenate, Attention
from keras.losses import MeanSquaredError, CategoricalCrossentropy
from keras.metrics import MeanAbsoluteError
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from random import randint


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
def gen_dataset(n_in, n_out, n_out_gen, n_features, n_samples):
    X_TIMES, PREVS_TIMES, y_TIMES = list(), list(), list()
    X_NOTES, PREVS_NOTES, y_NOTES = list(), list(), list()

    for _ in range(n_samples):
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
        target_in_times = pad_sequences(
            [t_src[:-3]],
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
            [nt_src[:-3]],
            maxlen=n_out,
            padding="pre",
            dtype="float",
        )[0]

        X_NOTES.append(to_categorical(source_notes, num_classes=n_features))
        PREVS_NOTES.append(to_categorical(target_in_notes, num_classes=n_features))
        y_NOTES.append(to_categorical(target_notes, num_classes=n_features))

    X_TIMES, PREVS_TIMES, y_TIMES = array(X_TIMES), array(PREVS_TIMES), array(y_TIMES)
    X_NOTES, PREVS_NOTES, y_NOTES = array(X_NOTES), array(PREVS_NOTES), array(y_NOTES)

    """ print(f"X_TIMES: {X_TIMES.shape}")
    print(f"PREVS_TIMES: {PREVS_TIMES.shape}")
    print(f"y_TIMES: {y_TIMES.shape}")

    print(f"X_NOTES: {X_NOTES.shape}")
    print(f"PREVS_NOTES: {PREVS_NOTES.shape}")
    print(f"y_NOTES: {y_NOTES.shape}") """

    return (X_TIMES, PREVS_TIMES, y_TIMES), (X_NOTES, PREVS_NOTES, y_NOTES)


# decode a one hot encoded string
def one_hot_decode_seq(encoded_seq):
    return [argmax(vector) for vector in encoded_seq]


def one_hot_decode(encoded):
    return argmax(encoded)


# returns train, inference_encoder and inference_decoder models
def define_models(n_input, n_output, n_features, n_units):
    encoder_inputs_times = Input(shape=(None, n_input), name="encoder_inputs_times")
    encoder_inputs_notes = Input(shape=(None, n_features), name="encoder_inputs_notes")

    encoder_times = Bidirectional(
        LSTM(n_units, return_state=True, name="encoder_lstm_times")
    )
    encoder_notes = Bidirectional(
        LSTM(n_units, return_state=True, name="encoder_lstm_notes")
    )

    (
        _,
        forward_state_h_times,
        forward_state_c_times,
        backward_state_h_times,
        backward_state_c_times,
    ) = encoder_times(encoder_inputs_times)

    (
        _,
        forward_state_h_notes,
        forward_state_c_notes,
        backward_state_h_notes,
        backward_state_c_notes,
    ) = encoder_notes(encoder_inputs_notes)

    state_h_times = Concatenate()(
        [
            forward_state_h_times,
            backward_state_h_times,
        ]
    )
    state_c_times = Concatenate()(
        [
            forward_state_c_times,
            backward_state_c_times,
        ]
    )
    state_h_notes = Concatenate()(
        [
            forward_state_h_notes,
            backward_state_h_notes,
        ]
    )
    state_c_notes = Concatenate()(
        [
            forward_state_c_notes,
            backward_state_c_notes,
        ]
    )

    # encoder_states_times = [state_h_times, state_c_times]
    # encoder_states_notes = [state_h_notes, state_c_notes]

    encoder_states = [
        state_h_times + state_h_notes,
        state_c_times + state_c_notes,
    ]

    # Define training decoder
    decoder_inputs_times = Input(shape=(None, n_input), name="decoder_inputs_times")
    decoder_inputs_notes = Input(shape=(None, n_features), name="decoder_inputs_notes")

    decoder_lstm_times = LSTM(
        n_units * 2,  # Bidirectional doubles the number of units
        return_sequences=True,
        return_state=True,
        name="decoder_lstm_times",
    )
    decoder_lstm_notes = LSTM(
        n_units * 2,  # Bidirectional doubles the number of units
        return_sequences=True,
        return_state=True,
        name="decoder_lstm_notes",
    )

    decoder_outputs_times, _, _ = decoder_lstm_times(
        decoder_inputs_times, initial_state=encoder_states
    )
    decoder_outputs_notes, _, _ = decoder_lstm_notes(
        decoder_inputs_notes, initial_state=encoder_states
    )

    decoder_dense_times = Dense(
        n_output, activation="linear", name="decoder_outputs_times"
    )
    decoder_outputs_times = decoder_dense_times(decoder_outputs_times)

    decoder_dense_notes = Dense(
        n_features, activation="softmax", name="decoder_outputs_notes"
    )
    decoder_outputs_notes = decoder_dense_notes(decoder_outputs_notes)

    inputs = [
        encoder_inputs_times,
        encoder_inputs_notes,
        decoder_inputs_times,
        decoder_inputs_notes,
    ]
    outputs = [decoder_outputs_times, decoder_outputs_notes]

    model = Model(inputs, outputs, name="model")

    # Define inference encoder
    encoder_model = Model(
        [encoder_inputs_times, encoder_inputs_notes],
        encoder_states,
        name="encoder",
    )

    # Define inference decoder
    decoder_state_input_h_times = Input(
        shape=(n_units * 2,), name="decoder_state_input_h_times"
    )  # Adjust shape for bidirectional
    decoder_state_input_c_times = Input(
        shape=(n_units * 2,), name="decoder_state_input_c_times"
    )
    decoder_state_input_h_notes = Input(
        shape=(n_units * 2,), name="decoder_state_input_h_notes"
    )  # Adjust shape for bidirectional
    decoder_state_input_c_notes = Input(
        shape=(n_units * 2,), name="decoder_state_input_c_notes"
    )

    decoder_states_inputs = [
        decoder_state_input_h_times + decoder_state_input_h_notes,
        decoder_state_input_c_times + decoder_state_input_c_notes,
    ]

    decoder_outputs_times, state_h_times, state_c_times = decoder_lstm_times(
        decoder_inputs_times, initial_state=decoder_states_inputs
    )
    decoder_outputs_notes, state_h_notes, state_c_notes = decoder_lstm_notes(
        decoder_inputs_notes, initial_state=decoder_states_inputs
    )

    decoder_states = [state_h_times + state_h_notes, state_c_times + state_c_notes]
    decoder_outputs_times = decoder_dense_times(decoder_outputs_times)
    decoder_outputs_notes = decoder_dense_notes(decoder_outputs_notes)

    decoder_model = Model(
        [decoder_inputs_times, decoder_inputs_notes] + decoder_states_inputs,
        [decoder_outputs_times, decoder_outputs_notes] + decoder_states,
        name="decoder",
    )

    # Plot model
    from keras.utils import plot_model

    plot_model(
        model,
        to_file="model_mix.png",
        show_layer_names=True,
        show_shapes=True,
    )
    plot_model(
        encoder_model,
        to_file="encoder_model_mix.png",
        show_layer_names=True,
        show_shapes=True,
    )
    plot_model(
        decoder_model,
        to_file="decoder_model_mix.png",
        show_layer_names=True,
        show_shapes=True,
    )
    model.summary()

    return model, encoder_model, decoder_model


# generate target given source sequence
def predict_sequence(infenc, infdec, source, n_steps):
    X_TIMES = source[0]
    X_NOTES = source[1]

    """ print(f"X_TIMES: {X_TIMES.shape}")
    print(f"X_NOTES: {X_NOTES.shape}") """

    # encode
    states = infenc.predict([X_TIMES, X_NOTES], verbose=0)

    # print(f"states: {len(states)}")

    # start of sequence input
    target_seq_times = array([generate_sequence(1, empty=True)])
    target_seq_notes = array(
        [
            to_categorical(
                generate_sequence_hot(1, n_features, empty=True), num_classes=n_features
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
        # store prediction

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
n_features = 50 + 1
n_steps_in = 10
n_steps_out = 10
n_steps_out_gen = 8
times_in_out = 2

# define model
train_model, encoder, decoder = define_models(
    times_in_out, times_in_out, n_features, 128
)
train_model.compile(
    optimizer=Adam(),  # "rmsprop",
    loss={
        "decoder_outputs_times": MeanSquaredError(),
        "decoder_outputs_notes": CategoricalCrossentropy(),
    },
    metrics=["accuracy"],
)
# generate training dataset
TIMES, NOTES = gen_dataset(n_steps_in, n_steps_out, n_steps_out_gen, n_features, 10000)

# (X_TIMES, PREVS_TIMES, y_TIMES)

X_TIMES, PREVS_TIMES, y_TIMES = TIMES
X_NOTES, PREVS_NOTES, y_NOTES = NOTES

for i in range(10):
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
    print()


# train model
train_model.fit(
    [X_TIMES, X_NOTES, PREVS_TIMES, PREVS_NOTES],
    [y_TIMES, y_NOTES],
    epochs=10,
)

# spot check some examples
for _ in range(10):
    TIMES, NOTES = gen_dataset(n_steps_in, n_steps_out, n_steps_out_gen, n_features, 1)

    X_TIMES, PREVS_TIMES, y_TIMES = TIMES
    X_NOTES, PREVS_NOTES, y_NOTES = NOTES
    predict = predict_sequence(encoder, decoder, [X_TIMES, X_NOTES], n_steps_out)

    print(f"X: ", end=" ")
    for j, x in enumerate(X_TIMES[0]):
        print(f"{x}({one_hot_decode(X_NOTES[0][j])})", end=" ")
    print()
    print(f"Prevs: ", end=" ")
    for j, prev in enumerate(PREVS_TIMES[0]):
        print(f"{prev}({one_hot_decode(PREVS_NOTES[0][j])})", end=" ")
    print()
    print(f"y: ", end=" ")
    for j, y_ in enumerate(y_TIMES[0]):
        print(f"{y_}({one_hot_decode(y_NOTES[0][j])})", end=" ")
    print()
    print(f"predict: ", end=" ")
    for j, y_ in enumerate(predict[0]):
        print(f"{y_}({one_hot_decode(predict[1][j])})", end=" ")

    print()
    print()
