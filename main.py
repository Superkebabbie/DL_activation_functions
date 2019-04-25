import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from termcolor import colored

if __name__ == '__main__':
    print(colored("TensorFLow version: ", 'blue') + tf.VERSION)
    print(colored("tf.keras version: ", 'blue') + tf.keras.__version__)

    #  Model definition
    print(colored("Creating model:", 'yellow'))
    model = Sequential()

    # Must define the input shape in the first layer of the neural network
    model.add(
        tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    # Take a look at the model summary
    model.summary()

    # Model compilation
    print(colored("\nModel compiling...", 'yellow'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print(colored("Compilation done!", 'yellow'))

    # Training model
    print(colored("Starting training of model:", 'yellow'))
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    # Further break training data into train / validation sets (# put 5000 into validation set
    # and keep remaining 55,000 for train)
    (x_train, x_valid) = x_train[5000:], x_train[:5000]
    (y_train, y_valid) = y_train[5000:], y_train[:5000]

    # Reshape input data from (28, 28) to (28, 28, 1)
    w, h = 28, 28
    x_train = x_train.reshape(x_train.shape[0], w, h, 1)
    x_valid = x_valid.reshape(x_valid.shape[0], w, h, 1)
    x_test = x_test.reshape(x_test.shape[0], w, h, 1)

    # One-hot encode the labels
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_valid = tf.keras.utils.to_categorical(y_valid, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose=1, save_best_only=True)

    model.fit(x_train,
              y_train,
              batch_size=64,
              epochs=10,
              validation_data=(x_valid, y_valid),
              callbacks=[checkpointer])

    print(colored("Training done!", 'yellow'))

    # Load the weights with the best validation accuracy
    model.load_weights('model.weights.best.hdf5')

    # Testing model
    score = model.evaluate(x_test, y_test, verbose=0)
    print('\n' + colored('Test accuracy: ', 'yellow'), score[1])
