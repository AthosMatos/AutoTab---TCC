def split(inputs, outputs):
    # Split the dataset into training, validation, and test sets
    SPLIT_RATIO = [0.7, 0.2, 0.1]
    num_examples = len(inputs)
    split_points = [int(ratio * num_examples) for ratio in SPLIT_RATIO]
    X_train = inputs[: split_points[0]]
    Y_train = outputs[: split_points[0]]
    X_val = inputs[split_points[0] : split_points[0] + split_points[1]]
    Y_val = outputs[split_points[0] : split_points[0] + split_points[1]]
    X_test = inputs[split_points[0] + split_points[1] :]
    Y_test = outputs[split_points[0] + split_points[1] :]

    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)
