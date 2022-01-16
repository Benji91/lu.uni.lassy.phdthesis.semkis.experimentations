import os, logging

from keras import models, layers, activations
from keras.optimizers import Adam
from utils import analysisutils, networkutils, configutils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class CNN:

    # def __init__(self, x_train, y_train, x_test, y_test, x_dev, y_dev):
    def __init__(self, x_train, x_dev, x_test, y_train, y_dev, y_test):
        """"
        Initialize the CNN class.
        """

        # Hyper-Parameters
        self.BATCH_SIZE = 16
        self.NUM_EPOCH = 2
        self.number_of_outputs = 20
        self.verbose = 1
        self.shuffling = True
        self.cnn = self.build_cnn()

        # Datasets
        self.x_train = x_train
        self.y_train = y_train
        self.x_dev = x_dev
        self.y_dev = y_dev
        self.x_test = x_test
        self.y_test = y_test

        # Compiler Configuration
        self.cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        networkutils.save_model(self.cnn, configutils.configSectionMap("FOLDERS")['cnn_path'] +
                                configutils.configSectionMap("CNN")['cnn_untrained'])

    def build_cnn(self, input_shape=(310, 330, 1), nb_filter=64):
        """
        Build a completely new CNN model.
        :return:
        """
        lyr_input = layers.Input(shape=input_shape)

        lyr_1_conv_1 = layers.Conv2D(nb_filter, (5, 5), padding='same')(lyr_input)
        lyr_1_con_2 = layers.Conv2D(nb_filter, (5, 5), padding='same')(lyr_1_conv_1)
        lyr_1_pool = layers.MaxPooling2D(pool_size=(2, 2))(lyr_1_con_2)
        lyr_1_act = layers.ReLU()(lyr_1_pool)

        lyr_2_conv_1 = layers.Conv2D(2 * nb_filter, (3, 3))(lyr_1_act)
        lyr_2_conv_2 = layers.Conv2D(2 * nb_filter, (3, 3))(lyr_2_conv_1)
        lyr_2_pool = layers.MaxPooling2D(pool_size=(2, 2))(lyr_2_conv_2)
        lyr_2_act = layers.ReLU()(lyr_2_pool)

        lyr_3_flat = layers.Flatten()(lyr_2_act)

        lyr_4_dense = layers.Dense(2 * nb_filter, activation="relu")(lyr_3_flat)
        lyr_4_dropout = layers.Dropout(rate=0.3)(lyr_4_dense)

        lyr_5_dense = layers.Dense(nb_filter, activation="relu")(lyr_4_dropout)
        lyr_5_dropout = layers.Dropout(rate=0.3)(lyr_5_dense)

        lyr_6_dense = layers.Dense(self.number_of_outputs, activation='sigmoid')(lyr_5_dropout)

        model = models.Model(inputs=lyr_input, outputs=lyr_6_dense)
        print(model.summary())
        return model

    def load_stored_cnn(self, cnn_name):
        """
        Load and build a trained CNN model from disk
        :return:
        """
        # Load the builted CNN
        self.cnn = networkutils.load_model(configutils.configSectionMap("FOLDERS")['cnn_path'] + cnn_name)
        # logging.debug(self.cnn.summary())
        self.cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, filename):
        """
        Train any CNN Model on a number of epochs
        :return:
        """
        # Train the neural network without data augmentation i obtained an accuracy of 0.98114
        history = self.cnn.fit(
            self.x_train, self.y_train,
            batch_size=self.BATCH_SIZE,
            epochs=self.NUM_EPOCH,
            validation_data=(self.x_dev, self.y_dev),
            verbose=self.verbose,
            shuffle=self.shuffling
        )

        networkutils.save_model(
            self.cnn,
            configutils.configSectionMap("FOLDERS")['cnn_path'] + filename
        )

        analysisutils.plot_cnn_training_curve(history)
        return history

    def evaluate_model(self, filename):
        """
        Used for Evaluating a CNN Model
        :param filename:
        """
        f = open(configutils.configSectionMap("FOLDERS")['cnn_performance_path'] + filename, "w+")
        phases = ["training", "validation", "testing"]
        for phase in phases:
            if phase == "training":
                score = self.cnn.evaluate(self.x_train, self.y_train, verbose=0)

            if phase == "validation":
                score = self.cnn.evaluate(self.x_dev, self.y_dev, verbose=0)

            if phase == "testing":
                score = self.cnn.evaluate(self.x_test, self.y_test, verbose=0)

            print(f"Evaluation of Model on {phase.capitalize()} dataset")
            print("%s: %.2f%%" % (self.cnn.metrics_names[1], score[1] * 100))
            print("%s: %f" % (self.cnn.metrics_names[0], score[0]))
            f.write(f"Evaluation of Model on {phase.capitalize()} Dataset\r\n")
            f.write("===============================================\r\n")
            f.write("%s: %.2f%%\r\n" % (self.cnn.metrics_names[1], score[1] * 100))
            f.write("%s: %f\r\n" % (self.cnn.metrics_names[0], score[0]))
            f.write("===============================================\r\n")
        f.close()

    def generate_confusion_matrices(self, num_classes, filename):
        # PREDICTION ON THE TRAINING DATASET
        analysisutils.generate_confusion_matrix(self.cnn, self.x_train, self.y_train, filename+"_training", num_classes, N=2)

        # PREDICTION ON THE VALIDATION DATASET
        analysisutils.generate_confusion_matrix(self.cnn, self.x_dev, self.y_dev, filename+"_development", num_classes, N=2)

        # PREDICTION ON THE TEST DATASET
        analysisutils.generate_confusion_matrix(self.cnn, self.x_test, self.y_test, filename+"_testing", num_classes, N=2)