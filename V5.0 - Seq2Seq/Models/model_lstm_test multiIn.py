from numpy import array
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Bidirectional, Concatenate
from keras.losses import MeanSquaredError
from keras.metrics import MeanAbsoluteError
from keras.optimizers import Adam
import numpy as np
from keras.preprocessing.sequence import pad_sequences


# generate a sequence of random integers
def generate_sequence(length, empty=False):
    if empty:
        return np.zeros((length, 2)).round(2).tolist()
    # random float between 0 and 5
    return np.random.uniform(0, 5, size=(length, 2)).round(2).tolist()


# prepare data for the LSTM
def gen_dataset(n_in, n_out, n_out_gen, n_samples):
    X, PREVS, y = list(), list(), list()
    for _ in range(n_samples):
        # generate source sequence
        source = generate_sequence(n_in)
        # define padded target sequence
        t_src = source[:n_out_gen]
        t_src.reverse()

        # target = [0 for _ in range(n_in - len(target))] + target  # std
        target = pad_sequences([t_src], maxlen=n_out, padding="pre", dtype="float")[0]

        # create padded input target sequence
        # target_in = generate_sequence(n_out, None, empty=True)
        # target_in = [[0, 0]] + target[:-1]  # std

        target_in = pad_sequences(
            [t_src[:-3]],
            maxlen=n_out,
            padding="pre",
            dtype="float",
        )[0]

        # pad target sequence to same length as input
        # target_in = [0 for _ in range(n_in - len(target_in))] + target_in

        """ print(f"target_in: {target_in}")
        print(f"target: {target}")
        print(f"source: {source}") """

        X.append(source)
        PREVS.append(target_in)
        y.append(target)
    X, PREVS, y = array(X), array(PREVS), array(y)

    """ X, PREVS, y = (
        expand_dims(X, axis=-1),
        expand_dims(PREVS, axis=-1),
        expand_dims(y, axis=-1),
    ) """
    return X, PREVS, y


# returns train, inference_encoder and inference_decoder models
def define_models(n_input, n_output, n_units):
    encoder_inputs = Input(shape=(None, n_input), name="encoder_inputs")
    encoder = Bidirectional(LSTM(n_units, return_state=True, name="encoder_lstm"))
    (
        _,
        forward_state_h,
        forward_state_c,
        backward_state_h,
        backward_state_c,
    ) = encoder(encoder_inputs)
    state_h = Concatenate()([forward_state_h, backward_state_h])
    state_c = Concatenate()([forward_state_c, backward_state_c])
    encoder_states = [state_h, state_c]

    # Define training decoder
    decoder_inputs = Input(shape=(None, n_input), name="decoder_inputs")
    decoder_lstm = LSTM(
        n_units * 2,  # Bidirectional doubles the number of units
        return_sequences=True,
        return_state=True,
        name="decoder_lstm",
    )
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

    decoder_dense = Dense(n_output, activation="linear", name="decoder_outputs")
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs, name="model")

    # Define inference encoder
    encoder_model = Model(encoder_inputs, encoder_states, name="encoder")

    # Define inference decoder
    decoder_state_input_h = Input(
        shape=(n_units * 2,), name="decoder_state_input_h"
    )  # Adjust shape for bidirectional
    decoder_state_input_c = Input(
        shape=(n_units * 2,), name="decoder_state_input_c"
    )  # Adjust shape for bidirectional
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs
    )
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states,
        name="decoder",
    )

    # Plot model
    from keras.utils import plot_model

    plot_model(
        model,
        to_file="model_lstm_test.png",
        show_layer_names=True,
    )

    model.summary()

    return model, encoder_model, decoder_model


# generate target given source sequence
def predict_sequence(infenc, infdec, source, n_steps):
    # encode
    state = infenc.predict(source, verbose=0)
    # start of sequence input
    target_seq = array([generate_sequence(1, empty=True)])
    # print(f"target_seq: {target_seq.shape}")
    # collect predictions
    output = list()
    for _ in range(n_steps):
        # predict next char
        yhat, h, c = infdec.predict([target_seq] + state, verbose=0)
        yhat = yhat.round(2)
        # store prediction
        output.append(yhat[0, 0, :])
        # update state
        state = [h, c]
        # update target sequence
        target_seq = yhat
        """ print(f"out:")
        for out in output:
            print(out, end=" ")
        print() """
        # print(one_hot_decode(output))
    return array(output)


# configure problem
# n_features = 50 + 1
n_steps_in = 10
n_steps_out = 10
n_steps_out_gen = 8
# define model
train_model, encoder, decoder = define_models(2, 2, 128)
train_model.compile(
    optimizer=Adam(),  # "rmsprop",
    loss=MeanSquaredError(),
    metrics=[MeanAbsoluteError(), "accuracy"],
)
# generate training dataset
X, PREVS, y = gen_dataset(n_steps_in, n_steps_out, n_steps_out_gen, 10000)
print(X.shape, PREVS.shape, y.shape)

for i in range(10):
    print(f"X1: ", end=" ")
    for x in X[i]:
        print(x, end=" ")
    print()
    print(f"X2: ", end=" ")
    for prev in PREVS[i]:
        print(prev, end=" ")
    print()
    print(f"y: ", end=" ")
    for y_ in y[i]:
        print(y_, end=" ")
    print()
    print()


exit()
# train model
train_model.fit([X, PREVS], y, epochs=3)

# spot check some examples
for _ in range(10):
    X, PREVS, y = gen_dataset(n_steps_in, n_steps_out, n_steps_out_gen, 1)
    predict = predict_sequence(encoder, decoder, X, n_steps_out)

    print(f"X: ", end="")
    for x in X[0]:
        print(x, end=" ")
    print()
    print(f"PREVS: ", end="")
    for prev in PREVS[0]:
        print(prev, end=" ")
    print()
    print(f"y: ", end="")
    for y_ in y[0]:
        print(y_, end=" ")
    print()
    print(f"predict: ", end="")
    for p in predict:
        print(p, end=" ")
    print()
    print()
