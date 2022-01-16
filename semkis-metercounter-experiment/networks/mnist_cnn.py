import os, logging

from keras.layers import Dense, Dropout, Flatten, Convolution2D, MaxPooling2D
from keras.models import Sequential
from utils import analysisutils, networkutils, configutils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class CNN:
    def __init__(self, x_train, y_train, x_test, y_test, x_dev, y_dev):
        """
        Initialize the CNN class.
        """
        self.model_name = ""

        # Configuration hyper-parameters
        self.epochs = 50
        self.batch_size = 64
        self.verbose = 1
        self.shuffling = True

        # Original datasets
        self.x_train = x_train
        self.y_train = y_train

        self.x_test = x_test
        # NOTE(gm): Why is a `test` variable used?
        test = y_test
        self.y_test = test

        self.x_val = x_dev
        self.y_val = y_dev

        # Initialising model
        self.model = Sequential()

    def build_cnn(self):
        """
        Build a completely new CNN model.
        :return:
        """
        self.model.add(Convolution2D(filters=64, kernel_size=(5, 5), padding='Same', activation='relu', input_shape=(28, 28, 1)))
        self.model.add(Convolution2D(filters=64, kernel_size=(5, 5), padding='Same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(rate=0.5))

        self.model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
        self.model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Dropout(rate=0.5))

        self.model.add(Flatten())
        self.model.add(Dense(128, activation="relu"))
        self.model.add(Dropout(rate=0.5))
        self.model.add(Dense(10, activation="softmax"))

        # Define the optimizer
        logging.debug(self.model.summary())
        self.model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])

        networkutils.save_model(self.model, configutils.configSectionMap("FOLDERS")['cnn_path'] +
                                configutils.configSectionMap("CNN")['cnn_untrained'])

    def build_trained_cnn(self):
        """
        Load and build a trained CNN model from disk
        :return:
        """
        cnn_path = configutils.configSectionMap("FOLDERS")['cnn_path']
        model_path = cnn_path + configutils.configSectionMap("CNN")['cnn_trained']
        self.model = networkutils.load_model(model_path)
        logging.debug(self.model.summary())
        self.model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])

    def build_untrained_cnn(self):
        """
        Load and build a untrained CNN model from disk
        :return:
        """
        self.model = networkutils.load_model(
            configutils.configSectionMap("FOLDERS")['cnn_path'] +
            configutils.configSectionMap("CNN")['cnn_untrained']
        )
        logging.debug(self.model.summary())
        self.model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])

    def train(self, epoch=50):
        """
        Train any CNN Model on a number of epochs
        :return:
        """
        # Train the neural network without data augmentation i obtained an accuracy of 0.98114
        self.epochs = epoch
        history = self.model.fit(
            self.x_train, self.y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(self.x_val, self.y_val),
            verbose=self.verbose,
            shuffle=self.shuffling
        )
        networkutils.save_model(
            self.model,
            configutils.configSectionMap("FOLDERS")['cnn_path'] + configutils.configSectionMap("CNN")['cnn_trained']
        )
        analysisutils.plot_cnn_training_curve(history)
        return history

    def evaluate_model(self, filename):
        """
        Used for Evaluating a CNN Model
        :param filename:
        """

        f = open(configutils.configSectionMap("FOLDERS")['cnn_performance_path'] +
                 filename + configutils.configSectionMap("CNN")['cnn_performance'], "w+")

        phases = ["training", "validation", "testing"]
        for phase in phases:
            score = self.model.evaluate(self.x_train, self.y_train, verbose=0)

            print(f"Evaluation of Model on {phase.capitalize()} dataset")
            print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))
            print("%s: %.2f%%" % (self.model.metrics_names[0], score[0] * 100))

            f.write(f"Evaluation of Model on {phase.capitalize()} Dataset\r\n")
            f.write("===============================================\r\n")
            f.write("%s: %.2f%%\r\n" % (self.model.metrics_names[1], score[1] * 100))
            f.write("%s: %.2f%%\r\n" % (self.model.metrics_names[0], score[0] * 100))
            f.write("===============================================\r\n")
        f.close()

    def generate_confusion_matrices(self):
        # PREDICTION ON THE TRAINING DATASET
        analysisutils.generate_confusion_matrix(self.model, self.x_train, self.y_train, "training", N=1)

        # PREDICTION ON THE VALIDATION DATASET
        analysisutils.generate_confusion_matrix(self.model, self.x_val, self.y_val, "validation", N=1)

        # PREDICTION ON THE TEST DATASET
        analysisutils.generate_confusion_matrix(self.model, self.x_test, self.y_test, "testing", N=1)
