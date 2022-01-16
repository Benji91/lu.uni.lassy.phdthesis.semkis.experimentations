import os, logging

from keras.layers import Dense, Dropout, Flatten, Convolution2D, MaxPooling2D
from keras.models import Sequential
from utils import analysisutils, networkutils, configutils, datautils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class CNN():

    def __init__(self, x_train, y_train, x_test, y_test, x_dev, y_dev):
        self.modelname = ""

        # Configuration Hyperparameters
        self.epochs = 50
        self.batch_size = 64
        self.verbose = 2
        self.shuffling = True

        # Original Datasets
        self.x_train = x_train
        self.y_train = y_train

        self.x_test = x_test
        self.y_test = y_test

        self.x_val = x_dev
        self.y_val = y_dev

        # Initialising model
        self.model = Sequential()

    def build_cnn(self):
        self.model.add(
            Convolution2D(filters=64, kernel_size=(5, 5), padding='Same', activation='relu', input_shape=(28, 28, 1)))
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

        #networkutils.save_model(self.model, configutils.configSectionMap("FOLDERS")['cnn_path'] +
         #                       configutils.configSectionMap("CNN")['cnn_untrained'])

    def build_trained_cnn(self):
        self.model = networkutils.load_model(configutils.configSectionMap("FOLDERS")['cnn_path'] +
                                             configutils.configSectionMap("CNN")['cnn_trained'])
        logging.debug(self.model.summary())
        self.model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])

    def build_untrained_cnn(self):
        self.model = networkutils.load_model(configutils.configSectionMap("FOLDERS")['cnn_path'] +
                                             configutils.configSectionMap("CNN")['cnn_untrained'])
        logging.debug(self.model.summary())
        self.model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])

    def train(self, filename, epoch=50):
        """
        Train the CNN
        :return:
        """
        # Train the neural network without data augmentation i obtained an accuracy of 0.98114
        self.epochs = epoch
        history = self.model.fit(self.x_train, self.y_train,
                                 batch_size=self.batch_size,
                                 epochs=self.epochs,
                                 validation_data=(self.x_val, self.y_val),
                                 verbose=self.verbose,
                                 shuffle=self.shuffling)
        networkutils.save_model(self.model, configutils.configSectionMap("FOLDERS")['cnn_path'] + filename + configutils.configSectionMap("CNN")['cnn_trained'])
        analysisutils.plot_cnn_training_curve(history)
        return history

    def evaluate_model(self, filename):
        """
        Used for Evaluating a CNN Model
        :param model:
        :param x_test:
        :param y_test:
        :param settype:
        :return:
        """

        f = open(configutils.configSectionMap("FOLDERS")['cnn_performance_path'] +
                 filename +configutils.configSectionMap("CNN")['cnn_performance'], "w+")

        print("Evaluation of Model on Training dataset")
        score = self.model.evaluate(self.x_train, self.y_train, verbose=0)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))
        print("%s: %.2f%%" % (self.model.metrics_names[0], score[0] * 100))

        f.write("Evaluation of Model on Training Dataset\r\n")
        f.write("===============================================\r\n")
        f.write("%s: %.2f%%\r\n" % (self.model.metrics_names[1], score[1] * 100))
        f.write("%s: %.2f%%\r\n" % (self.model.metrics_names[0], score[0] * 100))
        f.write("===============================================\r\n")

        print("Evaluation of Model on Validation dataset")
        score = self.model.evaluate(self.x_val, self.y_val, verbose=0)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))
        print("%s: %.2f%%" % (self.model.metrics_names[0], score[0] * 100))

        f.write("Evaluation of Model on Validation Dataset\r\n")
        f.write("===============================================\r\n")
        f.write("%s: %.2f%%\r\n" % (self.model.metrics_names[1], score[1] * 100))
        f.write("%s: %.2f%%\r\n" % (self.model.metrics_names[0], score[0] * 100))
        f.write("===============================================\r\n")

        print("Evaluation of Model on Testing dataset")
        score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))
        print("%s: %.2f%%" % (self.model.metrics_names[0], score[0] * 100))

        f.write("Evaluation of Model on Testing Dataset\r\n")
        f.write("===============================================\r\n")
        f.write("%s: %.2f%%\r\n" % (self.model.metrics_names[1], score[1] * 100))
        f.write("%s: %.2f%%\r\n" % (self.model.metrics_names[0], score[0] * 100))
        f.write("===============================================\r\n")
        f.close

    def generate_confusion_matrices(self):
        # PREDICTION ON THE TRAINING DATASET
        analysisutils.generate_confusion_matrix(self.model, self.x_train, self.y_train, "training")

        # PREDICTION ON THE VALIDATION DATASET
        analysisutils.generate_confusion_matrix(self.model, self.x_val, self.y_val, "validation")

        # PREDICTION ON THE TEST DATASET
        analysisutils.generate_confusion_matrix(self.model, self.x_test, self.y_test, "testing")
