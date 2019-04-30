import tensorflow as tf
import csv
import time
import random
import numpy as np

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from termcolor import colored


def create_model(activation='relu', optimizer='adam', verbose=0):
    """
    Creates a tf.keras.models.Sequential model using the parameters provided.
    :param activation: string describing the activation function to be used.
    :param optimizer: string describing the optimizer function to be used.
    :param verbose: controls the amount of logging done. 0 = no output, 1 is summarized output, 2 is detailed output.
    Defaults to 0.
    :return:
    """

    #  Model definition
    print(colored("Creating model:", 'yellow')) if verbose else None
    model = Sequential(name=activation+optimizer)

    # Must define the input shape in the first layer of the neural network
    model.add(
        tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same',
                               activation=activation, input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation=activation))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation=activation))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    # Take a look at the model summary
    model.summary() if verbose else None

    # Model compilation
    print(colored("\nModel compiling...", 'yellow')) if verbose else None
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    print(colored("Compilation done!", 'yellow')) if verbose else None
    return model


def train(model, x_train, y_train, x_valid, y_valid, iteration, verbose=0):
    """
    Trains and validates a model for the input presented.
    :param model: model to be trained.
    :param x_train: features of the training data.
    :param y_train: labels for the training data.
    :param x_valid: features of the validation data.
    :param y_valid: labels for the validation data.
    :param verbose: controls the amount of logging done. 0 = no output, 1 is summarized output, 2 is detailed output.
    Defaults to 0.
    :return: returns time to train as datetime object
    """

    # Training model
    print(colored("Starting training of model:", 'yellow')) if verbose else None

    # Reshape input data from (28, 28) to (28, 28, 1)
    w, h = 28, 28
    x_train = x_train.reshape(x_train.shape[0], w, h, 1)
    x_valid = x_valid.reshape(x_valid.shape[0], w, h, 1)

    # One-hot encode the labels
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_valid = tf.keras.utils.to_categorical(y_valid, 10)

    checkpointer = ModelCheckpoint(filepath=model.name+str(iteration)+'.hdf5', verbose=verbose, save_best_only=True)

    start = time.time()
    model.fit(x_train,
              y_train,
              batch_size=64,
              epochs=10,
              validation_data=(x_valid, y_valid),
              callbacks=[checkpointer],
              verbose=verbose)
    end = time.time()

    print(colored("Training done!", 'yellow')) if verbose else None

    return end - start


def test(model, x_test, y_test, iteration, verbose=0):
    """
    Will test the given model on the presented data.
    :param model: tf.keras.models.Sequential model.
    :param x_test: test image data.
    :param y_test: test labels.
    :param verbose: controls the amount of logging done. 0 = no output, 1 is summarized output, 2 is detailed output.
    Defaults to 0.
    :return: the accuracy score.
    """
    w, h = 28, 28
    x_test = x_test.reshape(x_test.shape[0], w, h, 1)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    # Load the weights with the best validation accuracy
    model.load_weights(model.name+str(iteration)+'.hdf5')

    # Testing model
    score = model.evaluate(x_test, y_test, verbose=0)
    print('\n' + colored('Test accuracy: ', 'yellow'), score[1]) if verbose else None
    return score[1]


def cross_validate(activation, optimizer, verbose=0):
    """
    Performs a cross-validation experiment on a newly created model using the desired parameters.
    :param activation:
    :param optimizer:
    :param verbose:
    :return:
    """
    print(colored('Collecting data for: ', 'yellow') + activation + " & " + optimizer)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    x_all = np.concatenate((x_train, x_test))
    y_all = np.concatenate((y_train, y_test))

    k = 10
    partition_size = len(x_all) // k  # 10-fold cross validation

    fold_indices = [i for i in range(len(x_all))]
    random.shuffle(fold_indices)  # disable for deterministic partitions
    fold_indices = fold_indices[:-(len(x_all) % k) if len(x_all) % k != 0 else len(x_all)]  # ensures even folds
    fold_indices = [fold_indices[n * partition_size:(n + 1) * partition_size] for n in range(k)]  # partition indexes

    accuracies = []
    times = []

    # Assign one fold validation, one test and all the rest train
    for test_fold in range(k):
        print('test_fold =', test_fold)
        x_test = [np.asarray(x_all[i]) for i in fold_indices[test_fold]]
        y_test = [np.asarray(y_all[i]) for i in fold_indices[test_fold]]

        rest = [x for x in range(k) if x != test_fold]
        dev_fold = rest[random.randint(0, len(rest) - 1)]
        x_valid = [np.asarray(x_all[i]) for i in fold_indices[dev_fold]]
        y_valid = [np.asarray(y_all[i]) for i in fold_indices[dev_fold]]

        rest = [x for x in rest if x != dev_fold]
        x_train = [np.asarray(x_all[i]) for i in [item for sublist in [fold_indices[f] for f in rest] for item in sublist]]
        y_train = [np.asarray(y_all[i]) for i in [item for sublist in [fold_indices[f] for f in rest] for item in sublist]]

        print("#X_train:%d\t#Y_train: %d\n#X_valid:%d\t#Y_valid: %d\n#X_test:%d\t#Y_test: %d" % (
            len(x_train), len(y_train), len(x_valid), len(y_valid), len(x_test), len(y_test))) if verbose else None

        # Create model, train and test it.
        model = create_model(activation, optimizer, verbose)
        time_to_train = train(model, np.asarray(x_train), np.asarray(y_train), np.asarray(x_valid), np.asarray(y_valid),
                              test_fold, verbose)
        accuracy = test(model, np.asarray(x_test), np.asarray(y_test), test_fold, verbose)

        times.append(time_to_train)
        accuracies.append(accuracy)

    return np.mean(accuracies), np.std(accuracies), np.mean(times), np.std(times)


def run():
    print(colored("TensorFLow version: ", 'blue') + tf.VERSION)
    print(colored("tf.keras version: ", 'blue') + tf.keras.__version__)

    activations = ['elu', 'relu', 'selu']
    optimizers = ['adadelta', 'adagrad', 'adam']

    # Iterate over the combinations
    results = [[(cross_validate(activation, optimizer, 1)) for optimizer in optimizers] for activation in activations]

    # Write results to file
    with open('results.cvs', mode='w') as results_csv:
        writer = csv.writer(results_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        writer.writerow([''] + optimizers)
        for row in range(len(results)):
            writer.writerow([activations[row]] + results[row])


if __name__ == '__main__':
    run()
