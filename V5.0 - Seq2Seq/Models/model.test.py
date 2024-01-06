from numpy import array, argmax, array_equal
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.losses import categorical_crossentropy, sparse_categorical_crossentropy
import numpy as np
from keras.preprocessing.sequence import pad_sequences


# generate a sequence of random integers
def generate_sequence(length, n_unique, empty=False):
    if empty:
        return [0 for _ in range(length)]
    return np.random.randint(1, n_unique, size=(length, 2)).tolist()


# prepare data for the LSTM
def gen_dataset(n_in, n_out, n_out_gen, cardinality, n_samples):
    X, PREVS, y = list(), list(), list()
    for _ in range(n_samples):
        # generate source sequence
        source = generate_sequence(n_in, cardinality)
        # define padded target sequence
        # print(f"source: {source}")

        target = source[: n_out_gen + 1]
        target.reverse()

        # print(f"target: {target}")

        # target = [0 for _ in range(n_in - len(target))] + target  # std
        # target = [0 for _ in range(n_out - len(target))] + target

        # create padded input target sequence
        # target_in = generate_sequence(n_out, None, empty=True)
        # target_in = [[0, 0]] + target[:-1]  # std

        target_in = pad_sequences([target[:-1]], maxlen=n_out, padding="pre")[0]

        # pad target sequence to same length as input
        # target_in = [0 for _ in range(n_in - len(target_in))] + target_in

        # print(f"target_in: {target_in}")

        # encode
        src_encoded = to_categorical(source, num_classes=cardinality)
        tar_encoded = to_categorical(target, num_classes=cardinality)
        tar2_encoded = to_categorical(target_in, num_classes=cardinality)
        # store
        X.append(src_encoded)
        PREVS.append(tar2_encoded)
        y.append(tar_encoded)
        """ X.append(source)
        PREVS.append(target_in) """
        # y.append(target)
    return array(X), array(PREVS), array(y)


# returns train, inference_encoder and inference_decoder models
def define_models(n_input, n_output, cardinality, n_units):
    # define training encoder
    encoder_times_inputs = Input(shape=(None, n_input), name="encoder_time_inputs")
    encoder_notes_inputs = Input(shape=(None, n_input), name="encoder_notes_inputs")

    encoder_times = LSTM(n_units, return_state=True, name="encoder_times")
    _, state_h_t, state_c_t = encoder_times(encoder_times_inputs)
    encoder_states_times = [state_h_t, state_c_t]

    encoder_notes = LSTM(n_units, return_state=True, name="encoder_notes")
    _, state_h_n, state_c_n = encoder_notes(encoder_notes_inputs)
    encoder_states_notes = [state_h_n, state_c_n]

    for i in range(2):
        print(f"encoder_states_times: {encoder_states_times[i]}")
        print(f"encoder_states_notes: {encoder_states_notes[i]}")

    """ # define training decoder
    decoder_inputs = Input(shape=(None, n_output), name="decoder_inputs")
    decoder_lstm = LSTM(
        n_units, return_sequences=True, return_state=True, name="decoder_lstm"
    )
    decoder_outputs, _, _ = decoder_lstm(
        decoder_inputs, initial_state=encoder_states_times
    )
    decoder_dense = Dense(cardinality, activation="softmax", name="decoder_outputs")
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_times_inputs, decoder_inputs], decoder_outputs, name="model")

    # define inference encoder
    encoder_model = Model(encoder_times_inputs, encoder_states_times, name="encoder")
    # define inference decoder
    decoder_state_input_h = Input(shape=(n_units,), name="decoder_state_input_h")
    decoder_state_input_c = Input(shape=(n_units,), name="decoder_state_input_c")
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h_t, state_c_t = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs
    )
    decoder_states = [state_h_t, state_c_t]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states,
        name="decoder",
    )

    # plot model
    from keras.utils import plot_model

    plot_model(
        model,
        to_file="model_lstm_test.png",
        show_shapes=True,
        show_layer_names=True,
    )

    model.summary()

    # return all models
    return model, encoder_model, decoder_model """


# generate target given source sequence
def predict_sequence(infenc, infdec, source, n_steps, cardinality):
    # encode
    state = infenc.predict(source, verbose=0)
    # start of sequence input
    target_seq = array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)

    # collect predictions
    output = list()
    for _ in range(n_steps):
        # predict next char
        # print(f"target_seq: {one_hot_decode(output)}")
        yhat, h, c = infdec.predict([target_seq] + state, verbose=0)
        # store prediction
        output.append(yhat[0, 0, :])
        # update state
        state = [h, c]
        # update target sequence
        target_seq = yhat

        # print(one_hot_decode(output))
    return array(output)


# decode a one hot encoded string
def one_hot_decode(encoded_seq):
    seqs = []
    for values in encoded_seq:
        seq = []
        for value in values:
            seq.append(argmax(value))
        seqs.append(seq)
    return seqs


# configure problem
n_features = 2
cardinality = 50 + 1
n_steps_in = 6
n_steps_out = 3
n_steps_out_gen = 2
# define model
define_models(n_features, n_features, cardinality, 128)
""" train_model, encoder, decoder = define_models(n_features, n_features, cardinality, 128)
train_model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)
# generate training dataset
X, PREVS, y = gen_dataset(n_steps_in, n_steps_out, n_steps_out_gen, cardinality, 100000)
print(X.shape, PREVS.shape, y.shape)
for i in range(10):
    print(
        f"X1: {one_hot_decode(X[i])} X2: {one_hot_decode(PREVS[i])} y: {one_hot_decode(y[i])}"
    )
 """
# train model

"""

train_model.fit([X, PREVS], y, epochs=1)



# evaluate LSTM
total, correct = 100, 0

for _ in range(total):
    X, PREVS, y = gen_dataset(n_steps_in, n_steps_out, n_steps_out_gen, n_features, 1)
    predict = predict_sequence(encoder, decoder, X, n_steps_out, n_features)
    if array_equal(one_hot_decode(y[0]), one_hot_decode(predict)):
        correct += 1
print("Accuracy: %.2f%%" % (float(correct) / float(total) * 100.0))
# spot check some examples
for _ in range(10):
    X, PREVS, y = gen_dataset(n_steps_in, n_steps_out, n_steps_out_gen, n_features, 1)
    predict = predict_sequence(encoder, decoder, X, n_steps_out, n_features)
    print(
        "X=%s y=%s, yhat=%s"
        % (one_hot_decode(X[0]), one_hot_decode(y[0]), one_hot_decode(predict))
    )

 """
