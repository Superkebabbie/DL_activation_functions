import tensorflow as tf
import csv
import time

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
    model = Sequential()

    # Must define the input shape in the first layer of the neural network
    model.add(
        tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation=activation,
                               input_shape=(28, 28, 1)))
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


def train(model, x_train, y_train, x_valid, y_valid, verbose=0):
    """
    Trains a model for the input presented.
    :param model: model to be trained.
    :param x_train: features of the training data.
    :param y_train: labels for the training data.
    :param x_valid: features of the validation data.
    :param y_valid: labels for the validation data.
    :param verbose: controls the amount of logging done. 0 = no output, 1 is summarized output, 2 is detailed output.
    Defaults to 0.
    :return: returns
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

    checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose=verbose, save_best_only=True)

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


def test(model, x_test, y_test, verbose=0):
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
    model.load_weights('model.weights.best.hdf5')

    # Testing model
    score = model.evaluate(x_test, y_test, verbose=0)
    print('\n' + colored('Test accuracy: ', 'yellow'), score[1]) if verbose else None
    return score[1]


def cross_validate(activation, optimizer):
    """
    Performs a cross-validation experiment on a newly created model using the desired parameters.
    :param activation:
    :param optimizer:
    :return:
    """
    print(colored('Collecting data for: ', 'yellow') + activation + " & " + optimizer)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    # TODO add the cross validation code here

    return mean_accuracy, std_accuracy, mean_time, std_time


def run():
    print(colored("TensorFLow version: ", 'blue') + tf.VERSION)
    print(colored("tf.keras version: ", 'blue') + tf.keras.__version__)

    activations = ['elu', 'relu', 'selu']
    optimizers = ['adadelta', 'adagrad', 'adam']

    # Iterate over the combinations
    results = [[(cross_validate(activation, optimizer)) for optimizer in optimizers] for activation in activations]

    # Write results to file
    with open('results.cvs', mode='w') as results_csv:
        writer = csv.writer(results_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        writer.writerow([''] + optimizers)
        for row in range(len(results)):
            writer.writerow([activations[row]] + results[row])


if __name__ == '__main__':
    run()
